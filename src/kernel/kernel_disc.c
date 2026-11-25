/*
 * kernel_disc.c - the game's files read straight from its disc image
 *
 * See kernel_disc.h for what it serves and why. The store is the disc's name
 * table, read once: one node per file and directory, keyed by the path below
 * the root ("" is the root, '\' separators, matched without case), each
 * directory's children kept together and in name order, as an XDVDFS
 * directory's tree reads in order. A handle is a real host event (manual
 * reset, signalled), as in the RAM cache (kernel_ramfs.c), so the bridge's
 * token table, a wait on the handle and CloseHandle all behave as they do
 * for a disk file; a table here maps it to the open file object.
 *
 * Reads go to the image at the file's offset. The node table never changes
 * after the image is opened, so a read copies what it needs under the lock
 * and reads outside it: two threads streaming two files do not wait on each
 * other.
 */
#include "kernel.h"
#include "kernel_disc.h"
#include "guest_det_device.h"
#include "recomp_env.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifndef STATUS_NOT_A_DIRECTORY
#define STATUS_NOT_A_DIRECTORY          ((NTSTATUS)0xC0000103L)
#endif
#ifndef STATUS_BUFFER_TOO_SMALL
#define STATUS_BUFFER_TOO_SMALL         ((NTSTATUS)0xC0000023L)
#endif
#ifndef STATUS_FILE_IS_A_DIRECTORY
#define STATUS_FILE_IS_A_DIRECTORY      ((NTSTATUS)0xC00000BAL)
#endif
#ifndef STATUS_MEDIA_WRITE_PROTECTED
#define STATUS_MEDIA_WRITE_PROTECTED    ((NTSTATUS)0xC00000A2L)
#endif

#define DISC_SECTOR      2048ull
#define DISC_PATH        512
#define DISC_MAX_NODES   200000u
#define DISC_MAX_DEPTH   32

/* NtCreateFile's IoStatusBlock->Information. */
#define DISC_OPENED      1

extern uint32_t g_xbox_last_file_error;      /* kernel_file.c */

static const char s_magic[20] = { 'M','I','C','R','O','S','O','F','T','*',
                                  'X','B','O','X','*','M','E','D','I','A' };

/* Where a dump may put the game partition: the same list as xdvdfs.py. */
static const uint64_t s_bases[] = {
    0x00000000ull, 0x0000FD90ull, 0x00030600ull, 0x0FD90000ull, 0x18300000ull,
};

typedef struct disc_node {
    char     *path;
    uint32_t  parent;           /* index of the directory holding it          */
    int       is_dir;
    uint64_t  offset;           /* files: byte offset in the image            */
    uint32_t  size;             /* files: bytes; directories: table bytes     */
} disc_node;

typedef struct disc_file {
    uint32_t  node;
    int       refs;             /* handles: NtDuplicateObject shares one      */
    uint64_t  pos;
    int       enum_started;
    int       enum_done;
    uint32_t  enum_next;        /* next node index to look at                 */
    char      pattern[64];
} disc_file;

typedef struct {
    HANDLE     h;
    disc_file *f;
} disc_handle;

static SRWLOCK      s_lock = SRWLOCK_INIT;
static volatile int s_active;
static int          s_env_checked;
static int          s_trace = -1;           /* RECOMP_DISC_TRACE: -1 not read */
static char        *s_image_path;
static uint64_t     s_base;
static uint64_t     s_image_size;
static LONGLONG     s_volume_time;
static disc_node   *s_nodes;
static uint32_t     s_node_count, s_node_cap;
static disc_handle *s_handles;
static size_t       s_handle_count, s_handle_cap;

#if defined(_WIN32)
static FILE        *s_fp;
static SRWLOCK      s_io_lock = SRWLOCK_INIT;
#else
static int          s_fd = -1;
#endif

/* ---- the image ---------------------------------------------------------- */

static int image_read(uint64_t off, void *buf, size_t len, size_t *got)
{
#if defined(_WIN32)
    size_t n;
    AcquireSRWLockExclusive(&s_io_lock);
    if (_fseeki64(s_fp, (__int64)off, SEEK_SET) != 0) {
        ReleaseSRWLockExclusive(&s_io_lock);
        return 0;
    }
    n = fread(buf, 1, len, s_fp);
    ReleaseSRWLockExclusive(&s_io_lock);
    *got = n;
    return n == len || !ferror(s_fp);
#else
    size_t done = 0;
    while (done < len) {
        ssize_t n = pread(s_fd, (char *)buf + done, len - done, (off_t)(off + done));
        if (n < 0)
            return 0;
        if (n == 0)
            break;
        done += (size_t)n;
    }
    *got = done;
    return 1;
#endif
}

static int path_eq(const char *a, const char *b)
{
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b))
            return 0;
        a++, b++;
    }
    return *a == *b;
}

static int name_cmp(const void *pa, const void *pb)
{
    const char *a = ((const disc_node *)pa)->path, *b = ((const disc_node *)pb)->path;
    const char *la = strrchr(a, '\\'), *lb = strrchr(b, '\\');
    a = la ? la + 1 : a;
    b = lb ? lb + 1 : b;
    for (;; a++, b++) {
        int ca = toupper((unsigned char)*a), cb = toupper((unsigned char)*b);
        if (ca != cb || !ca)
            return ca - cb;
    }
}

static int add_node(const char *path, uint32_t parent, int is_dir,
                    uint64_t offset, uint32_t size)
{
    disc_node *n;
    if (s_node_count >= DISC_MAX_NODES)
        return 0;
    if (s_node_count == s_node_cap) {
        uint32_t cap = s_node_cap ? s_node_cap * 2 : 256;
        disc_node *grown = (disc_node *)realloc(s_nodes, cap * sizeof *grown);
        if (!grown)
            return 0;
        s_nodes = grown;
        s_node_cap = cap;
    }
    n = &s_nodes[s_node_count];
    n->path = (char *)malloc(strlen(path) + 1);
    if (!n->path)
        return 0;
    strcpy(n->path, path);
    n->parent = parent;
    n->is_dir = is_dir;
    n->offset = offset;
    n->size = size;
    s_node_count++;
    return 1;
}

/* One directory's entries, appended in name order, then each subdirectory's.
 * Entries are walked as laid out, sector by sector, not by the tree's links:
 * a padded or damaged link would otherwise drop a subtree without a word
 * (tools/xiso/xdvdfs.py reads them the same way). */
static int read_dir(uint32_t dir, uint32_t sector, uint32_t size, int depth)
{
    uint8_t *table;
    size_t got, at;
    uint32_t first = s_node_count, i, end;

    if (depth > DISC_MAX_DEPTH)
        return 0;
    if (size == 0)
        return 1;
    table = (uint8_t *)malloc(size);
    if (!table)
        return 0;
    if (!image_read(s_base + sector * DISC_SECTOR, table, size, &got) || got != size) {
        free(table);
        return 0;
    }
    for (at = 0; at < size; at += DISC_SECTOR) {
        size_t off = 0, chunk = size - at < DISC_SECTOR ? size - at : DISC_SECTOR;
        const uint8_t *s = table + at;
        while (off + 0x0E <= chunk) {
            uint16_t l = (uint16_t)(s[off] | s[off + 1] << 8);
            uint16_t r = (uint16_t)(s[off + 2] | s[off + 3] << 8);
            uint32_t sec = (uint32_t)s[off + 4] | (uint32_t)s[off + 5] << 8
                         | (uint32_t)s[off + 6] << 16 | (uint32_t)s[off + 7] << 24;
            uint32_t sz = (uint32_t)s[off + 8] | (uint32_t)s[off + 9] << 8
                        | (uint32_t)s[off + 10] << 16 | (uint32_t)s[off + 11] << 24;
            uint8_t attr = s[off + 12], nlen = s[off + 13];
            char path[DISC_PATH];
            int k;

            if ((l == 0xFFFF && r == 0xFFFF) || nlen == 0 || off + 0x0E + nlen > chunk)
                break;
            if (dir == 0)
                k = snprintf(path, sizeof path, "%.*s", nlen, (const char *)s + off + 0x0E);
            else
                k = snprintf(path, sizeof path, "%s\\%.*s", s_nodes[dir].path, nlen,
                             (const char *)s + off + 0x0E);
            if (k < 0 || (size_t)k >= sizeof path) {
                free(table);
                return 0;
            }
            if (!add_node(path, dir, (attr & 0x10) != 0,
                          s_base + (uint64_t)sec * DISC_SECTOR, sz)) {
                free(table);
                return 0;
            }
            /* A directory keeps its table's sector until it is read below. */
            if (attr & 0x10)
                s_nodes[s_node_count - 1].offset = sec;
            off += (0x0E + nlen + 3u) & ~3u;
        }
    }
    free(table);
    end = s_node_count;
    qsort(s_nodes + first, end - first, sizeof *s_nodes, name_cmp);
    for (i = first; i < end; i++) {
        if (s_nodes[i].is_dir) {
            uint32_t sec = (uint32_t)s_nodes[i].offset;
            s_nodes[i].offset = 0;
            if (!read_dir(i, sec, s_nodes[i].size, depth + 1))
                return 0;
        }
    }
    return 1;
}

static void close_image(void)
{
    uint32_t i;
    for (i = 0; i < s_node_count; i++)
        free(s_nodes[i].path);
    free(s_nodes);
    s_nodes = NULL;
    s_node_count = s_node_cap = 0;
#if defined(_WIN32)
    if (s_fp)
        fclose(s_fp);
    s_fp = NULL;
#else
    if (s_fd >= 0)
        close(s_fd);
    s_fd = -1;
#endif
}

int xbox_DiscImageOpen(const char *path, char *err, size_t err_n)
{
    uint8_t desc[2048];
    size_t got, i;
    uint32_t root_sector = 0, root_size = 0;
    int found = 0;

    if (err && err_n)
        err[0] = '\0';
    if (s_active || s_image_path) {
        if (err)
            snprintf(err, err_n, "a disc image is already open (%s)",
                     s_image_path ? s_image_path : "?");
        return 0;
    }
    if (!path || !path[0]) {
        if (err)
            snprintf(err, err_n, "no disc image path");
        return 0;
    }
#if defined(_WIN32)
    {
        WCHAR wide[MAX_PATH * 2];
        if (!MultiByteToWideChar(CP_UTF8, 0, path, -1, wide, MAX_PATH * 2)
                || !(s_fp = _wfopen(wide, L"rb"))) {
            if (err)
                snprintf(err, err_n, "cannot open %s", path);
            return 0;
        }
        _fseeki64(s_fp, 0, SEEK_END);
        s_image_size = (uint64_t)_ftelli64(s_fp);
    }
#else
    {
        struct stat stt;
        s_fd = open(path, O_RDONLY);
        if (s_fd < 0 || fstat(s_fd, &stt) != 0) {
            if (err)
                snprintf(err, err_n, "cannot open %s", path);
            close_image();
            return 0;
        }
        s_image_size = (uint64_t)stt.st_size;
    }
#endif
    for (i = 0; i < sizeof s_bases / sizeof s_bases[0]; i++) {
        uint64_t at = s_bases[i] + 32 * DISC_SECTOR;
        if (at + sizeof desc > s_image_size)
            continue;
        if (image_read(at, desc, sizeof desc, &got) && got == sizeof desc
                && !memcmp(desc, s_magic, sizeof s_magic)) {
            s_base = s_bases[i];
            root_sector = (uint32_t)desc[0x14] | (uint32_t)desc[0x15] << 8
                        | (uint32_t)desc[0x16] << 16 | (uint32_t)desc[0x17] << 24;
            root_size = (uint32_t)desc[0x18] | (uint32_t)desc[0x19] << 8
                      | (uint32_t)desc[0x1A] << 16 | (uint32_t)desc[0x1B] << 24;
            memcpy(&s_volume_time, desc + 0x1C, sizeof s_volume_time);
            found = 1;
            break;
        }
    }
    if (!found) {
        if (err)
            snprintf(err, err_n, "%s is not an Xbox disc image (no XDVDFS volume "
                     "descriptor at any known partition offset)", path);
        close_image();
        return 0;
    }
    if (!add_node("", 0, 1, 0, root_size) || !read_dir(0, root_sector, root_size, 0)) {
        if (err)
            snprintf(err, err_n, "%s: the directory tables could not be read "
                     "(truncated or damaged image)", path);
        close_image();
        return 0;
    }
    s_image_path = (char *)malloc(strlen(path) + 1);
    if (s_image_path)
        strcpy(s_image_path, path);
    s_active = 1;
    fprintf(stderr, "  [FILE] disc image: %s, game partition at 0x%llX, %u names; "
            "the game files are read from it\n", path, (unsigned long long)s_base,
            s_node_count - 1);
    fflush(stderr);
    return 1;
}

const char *xbox_DiscImagePath(void)
{
    return s_active ? s_image_path : NULL;
}

int xbox_disc_active(void)
{
    if (!s_env_checked) {
        s_env_checked = 1;
        if (!s_active) {
            const char *p = getenv("RECOMP_DISC_IMAGE");
            char err[512];
            if (p && p[0] && !xbox_DiscImageOpen(p, err, sizeof err)) {
                fprintf(stderr, "  [FILE] RECOMP_DISC_IMAGE: %s\n", err);
                fflush(stderr);
            }
        }
    }
    return s_active;
}

static int find_node(const char *path, uint32_t *out)
{
    uint32_t i;
    for (i = 0; i < s_node_count; i++)
        if (path_eq(s_nodes[i].path, path)) {
            *out = i;
            return 1;
        }
    return 0;
}

int xbox_DiscReadFile(const char *rel, void **data, size_t *size)
{
    uint32_t i;
    void *buf;
    size_t got;

    *data = NULL;
    *size = 0;
    if (!xbox_disc_active() || !find_node(rel, &i) || s_nodes[i].is_dir)
        return 0;
    buf = malloc(s_nodes[i].size ? s_nodes[i].size : 1);
    if (!buf)
        return 0;
    if (!image_read(s_nodes[i].offset, buf, s_nodes[i].size, &got) || got != s_nodes[i].size) {
        free(buf);
        return 0;
    }
    *data = buf;
    *size = got;
    return 1;
}

/* ---- names -------------------------------------------------------------- */

/* '/' to '\', no leading, trailing or doubled separators. */
static int normalize(const char *in, char *out, size_t n)
{
    size_t o = 0;
    for (; *in; in++) {
        char c = *in == '/' ? '\\' : *in;
        if (c == '\\' && (o == 0 || out[o - 1] == '\\'))
            continue;
        if (o + 1 >= n)
            return 0;
        out[o++] = c;
    }
    while (o && out[o - 1] == '\\')
        o--;
    out[o] = '\0';
    return 1;
}

static int is_relative(const char *name)
{
    return name[0] != '\\' && name[0] != '/' && !(name[0] && name[1] == ':');
}

static const char *leaf_of(const char *path)
{
    const char *cut = strrchr(path, '\\');
    return cut ? cut + 1 : path;
}

/* Case-insensitive '*' and '?' match, as kernel_ramfs.c's. */
static int wild(const char *p, const char *s)
{
    for (; *p; p++, s++) {
        if (*p == '*') {
            while (p[1] == '*')
                p++;
            if (!p[1])
                return 1;
            for (; *s; s++)
                if (wild(p + 1, s))
                    return 1;
            return 0;
        }
        if (!*s)
            return 0;
        if (*p != '?' && tolower((unsigned char)*p) != tolower((unsigned char)*s))
            return 0;
    }
    return *s == '\0';
}

/* ---- handles ------------------------------------------------------------ */

static disc_file *find_file(HANDLE h, size_t *slot)
{
    size_t i;
    if (!h || h == INVALID_HANDLE_VALUE)
        return NULL;
    for (i = 0; i < s_handle_count; i++) {
        if (s_handles[i].h == h) {
            if (slot)
                *slot = i;
            return s_handles[i].f;
        }
    }
    return NULL;
}

static HANDLE add_handle(disc_file *f)
{
    HANDLE h;
    if (s_handle_count == s_handle_cap) {
        size_t cap = s_handle_cap ? s_handle_cap * 2 : 32;
        disc_handle *grown = (disc_handle *)realloc(s_handles, cap * sizeof *grown);
        if (!grown)
            return NULL;
        s_handles = grown;
        s_handle_cap = cap;
    }
    /* Signalled and manual reset: every request completes before it returns. */
    h = CreateEventW(NULL, TRUE, TRUE, NULL);
    if (!h)
        return NULL;
    s_handles[s_handle_count].h = h;
    s_handles[s_handle_count].f = f;
    s_handle_count++;
    f->refs++;
    return h;
}

static void release_file(disc_file *f)
{
    if (--f->refs <= 0)
        free(f);
}

/* RECOMP_DISC_TRACE=1: a line for each guest name the image serves, and for
 * each game-tree name it leaves to the disk (not on the disc, or a write). */
static int tracing(void)
{
    if (s_trace < 0) {
        const char *v = getenv("RECOMP_DISC_TRACE");
        s_trace = v && v[0] && strcmp(v, "0") != 0;
    }
    return s_trace;
}

static void trace(const char *what, PXBOX_OBJECT_ATTRIBUTES oa)
{
    if (tracing() && oa && oa->ObjectName && oa->ObjectName->Buffer) {
        fprintf(stderr, "  [DISC] %s %.*s\n", what, (int)oa->ObjectName->Length,
                oa->ObjectName->Buffer);
        fflush(stderr);
    }
}

/* 1 (disc root) or 3 (another game-tree root) with the node, 0 to leave the
 * name to the disk: not in the game tree, a bare device, or not on the disc.
 * Called without the lock for the path table (it takes the symbolic-link
 * table's), then with it for a RootDirectory. */
static int resolve(PXBOX_OBJECT_ATTRIBUTES oa, uint32_t *node)
{
    const char *name;
    char rest[DISC_PATH], rel[DISC_PATH];
    int r;

    if (!oa || !oa->ObjectName || !oa->ObjectName->Buffer)
        return 0;
    name = oa->ObjectName->Buffer;
    if (oa->RootDirectory && is_relative(name)) {
        disc_file *dir;
        char joined[DISC_PATH * 2];
        AcquireSRWLockExclusive(&s_lock);
        dir = find_file(oa->RootDirectory, NULL);
        r = 0;
        if (dir) {
            snprintf(joined, sizeof joined, "%s\\%s", s_nodes[dir->node].path, name);
            r = normalize(joined, rel, sizeof rel) && find_node(rel, node) ? 1 : 0;
        }
        ReleaseSRWLockExclusive(&s_lock);
        return r;
    }
    r = xbox_path_game_rest(name, rest, sizeof rest);
    if (r != 1 && r != 3)
        return 0;
    if (!normalize(rest, rel, sizeof rel) || !find_node(rel, node)) {
        trace("not on the disc:", oa);
        return 0;
    }
    return r;
}

static LONGLONG file_time(void)
{
    return xbox_GuestDetOn() ? XBOX_DET_FILETIME : s_volume_time;
}

static void fill_net(const disc_node *nd, PXBOX_FILE_NETWORK_OPEN_INFORMATION info)
{
    info->CreationTime.QuadPart = info->LastAccessTime.QuadPart =
        info->LastWriteTime.QuadPart = info->ChangeTime.QuadPart = file_time();
    info->EndOfFile.QuadPart = nd->is_dir ? 0 : (LONGLONG)nd->size;
    info->AllocationSize.QuadPart = nd->is_dir ? 0
        : (LONGLONG)((nd->size + DISC_SECTOR - 1) / DISC_SECTOR * DISC_SECTOR);
    /* A DVD's files are read-only, as XAPI reports them on a console. */
    info->FileAttributes = nd->is_dir
        ? XBOX_FILE_ATTRIBUTE_DIRECTORY | XBOX_FILE_ATTRIBUTE_READONLY
        : XBOX_FILE_ATTRIBUTE_READONLY;
}

static NTSTATUS fail_open(PXBOX_IO_STATUS_BLOCK ios, NTSTATUS st, uint32_t err)
{
    g_xbox_last_file_error = err;
    if (ios) {
        ios->Status = st;
        ios->Information = 0;
    }
    return st;
}

/* ---- the Nt*File entry points ------------------------------------------- */

int xbox_disc_create(PHANDLE out, ACCESS_MASK access, PXBOX_OBJECT_ATTRIBUTES oa,
                     PXBOX_IO_STATUS_BLOCK ios, ULONG disposition, ULONG options,
                     NTSTATUS *st)
{
    uint32_t node;
    int root, want_dir = (options & XBOX_FILE_DIRECTORY_FILE) != 0;
    int write = (access & (XBOX_GENERIC_WRITE | XBOX_GENERIC_ALL |
                           XBOX_FILE_WRITE_DATA | XBOX_FILE_APPEND_DATA)) != 0;
    int changes = write || (disposition != XBOX_FILE_OPEN && disposition != XBOX_FILE_OPEN_IF);
    const disc_node *n;
    disc_file *f;

    root = resolve(oa, &node);
    if (!root)
        return 0;
    /* Outside D: and CdRom0 the game dir stands in for a hard disk, so a
     * write there goes on to it as it did before; on the disc it fails, as
     * on a DVD. */
    if (changes && root == 3) {
        trace("write, to the disk:", oa);
        return 0;
    }
    if (!out) {
        *st = STATUS_INVALID_PARAMETER;
        return 1;
    }
    if (g_xbox_path_hook && !oa->RootDirectory)
        g_xbox_path_hook(oa->ObjectName->Buffer);

    AcquireSRWLockExclusive(&s_lock);
    n = &s_nodes[node];
    if (want_dir && !n->is_dir) {
        *st = fail_open(ios, STATUS_NOT_A_DIRECTORY, 267);
        goto out;
    }
    if (n->is_dir && (options & XBOX_FILE_NON_DIRECTORY_FILE)) {
        *st = fail_open(ios, STATUS_FILE_IS_A_DIRECTORY, 5);
        goto out;
    }
    if (disposition == XBOX_FILE_CREATE) {
        *st = fail_open(ios, STATUS_OBJECT_NAME_COLLISION, 80);
        goto out;
    }
    if (changes) {
        *st = fail_open(ios, n->is_dir && !want_dir ? STATUS_FILE_IS_A_DIRECTORY
                                                     : STATUS_MEDIA_WRITE_PROTECTED, 19);
        goto out;
    }
    f = (disc_file *)calloc(1, sizeof *f);
    if (!f) {
        *st = fail_open(ios, STATUS_NO_MEMORY, 8);
        goto out;
    }
    f->node = node;
    *out = add_handle(f);
    if (!*out) {
        free(f);
        *st = fail_open(ios, STATUS_INSUFFICIENT_RESOURCES, 4);
        goto out;
    }
    if (ios) {
        ios->Status = STATUS_SUCCESS;
        ios->Information = DISC_OPENED;
    }
    trace("open", oa);
    *st = STATUS_SUCCESS;
out:
    ReleaseSRWLockExclusive(&s_lock);
    return 1;
}

int xbox_disc_read(HANDLE h, HANDLE event, PXBOX_IO_STATUS_BLOCK ios,
                   PVOID buf, ULONG len, PLARGE_INTEGER off, NTSTATUS *st)
{
    disc_file *f;
    uint64_t pos, want = 0, at = 0;
    size_t got = 0;
    int is_dir, ok = 1;

    AcquireSRWLockExclusive(&s_lock);
    f = find_file(h, NULL);
    if (!f) {
        ReleaseSRWLockExclusive(&s_lock);
        return 0;
    }
    is_dir = s_nodes[f->node].is_dir;
    pos = off && off->QuadPart >= 0 ? (uint64_t)off->QuadPart : f->pos;
    if (!is_dir && pos < s_nodes[f->node].size) {
        want = s_nodes[f->node].size - pos;
        if (want > len)
            want = len;
        at = s_nodes[f->node].offset + pos;
    }
    ReleaseSRWLockExclusive(&s_lock);

    if (!ios) {
        *st = STATUS_INVALID_PARAMETER;
        return 1;
    }
    if (is_dir) {
        ios->Status = *st = STATUS_INVALID_DEVICE_REQUEST;
        ios->Information = 0;
        return 1;
    }
    if (want)
        ok = image_read(at, buf, (size_t)want, &got);
    if (!ok) {
        ios->Status = *st = STATUS_UNSUCCESSFUL;
        ios->Information = 0;
        return 1;
    }

    AcquireSRWLockExclusive(&s_lock);
    if (find_file(h, NULL) == f)
        f->pos = pos + got;
    ReleaseSRWLockExclusive(&s_lock);
    ios->Information = (ULONG_PTR)got;
    if (got == 0 && len > 0) {
        ios->Status = *st = STATUS_END_OF_FILE;
    } else {
        ios->Status = *st = STATUS_SUCCESS;
        if (event)
            SetEvent(event);
    }
    return 1;
}

int xbox_disc_write(HANDLE h, PXBOX_IO_STATUS_BLOCK ios, NTSTATUS *st)
{
    int mine;
    AcquireSRWLockExclusive(&s_lock);
    mine = find_file(h, NULL) != NULL;
    ReleaseSRWLockExclusive(&s_lock);
    if (!mine)
        return 0;
    if (ios) {
        ios->Status = STATUS_ACCESS_DENIED;
        ios->Information = 0;
    }
    *st = STATUS_ACCESS_DENIED;
    return 1;
}

int xbox_disc_close(HANDLE h)
{
    disc_file *f;
    size_t slot;

    AcquireSRWLockExclusive(&s_lock);
    f = find_file(h, &slot);
    if (f) {
        /* Out of the table before CloseHandle: the host can hand the same
         * value straight back for the next handle, which may be a disk one. */
        s_handles[slot] = s_handles[--s_handle_count];
        release_file(f);
    }
    ReleaseSRWLockExclusive(&s_lock);
    if (f)
        CloseHandle(h);
    return f != NULL;
}

int xbox_disc_delete(PXBOX_OBJECT_ATTRIBUTES oa, NTSTATUS *st)
{
    uint32_t node;
    if (resolve(oa, &node) != 1)
        return 0;
    *st = STATUS_MEDIA_WRITE_PROTECTED;
    return 1;
}

int xbox_disc_query_info(HANDLE h, PXBOX_IO_STATUS_BLOCK ios, PVOID info,
                         ULONG len, XBOX_FILE_INFORMATION_CLASS cls, NTSTATUS *st)
{
    disc_file *f;
    const disc_node *n;

    AcquireSRWLockExclusive(&s_lock);
    f = find_file(h, NULL);
    if (!f) {
        ReleaseSRWLockExclusive(&s_lock);
        return 0;
    }
    n = &s_nodes[f->node];
    if (!ios || !info) {
        *st = STATUS_INVALID_PARAMETER;
        goto out;
    }
    switch (cls) {
        case XboxFileBasicInformation: {
            PXBOX_FILE_BASIC_INFORMATION b = (PXBOX_FILE_BASIC_INFORMATION)info;
            XBOX_FILE_NETWORK_OPEN_INFORMATION net;
            fill_net(n, &net);
            b->CreationTime = net.CreationTime;
            b->LastAccessTime = net.LastAccessTime;
            b->LastWriteTime = net.LastWriteTime;
            b->ChangeTime = net.ChangeTime;
            b->FileAttributes = net.FileAttributes;
            ios->Information = sizeof *b;
            break;
        }
        case XboxFileStandardInformation: {
            PXBOX_FILE_STANDARD_INFORMATION s = (PXBOX_FILE_STANDARD_INFORMATION)info;
            XBOX_FILE_NETWORK_OPEN_INFORMATION net;
            fill_net(n, &net);
            s->EndOfFile = net.EndOfFile;
            s->AllocationSize = net.AllocationSize;
            s->NumberOfLinks = 1;
            s->DeletePending = FALSE;
            s->Directory = n->is_dir ? TRUE : FALSE;
            ios->Information = sizeof *s;
            break;
        }
        case XboxFilePositionInformation:
            ((PXBOX_FILE_POSITION_INFORMATION)info)->CurrentByteOffset.QuadPart =
                (LONGLONG)f->pos;
            ios->Information = sizeof(XBOX_FILE_POSITION_INFORMATION);
            break;
        case XboxFileInternalInformation:
            if (len < sizeof(XBOX_FILE_INTERNAL_INFORMATION)) {
                *st = STATUS_INFO_LENGTH_MISMATCH;
                ios->Status = *st;
                ios->Information = 0;
                goto out;
            }
            /* The node index names the file for as long as the image is open. */
            ((PXBOX_FILE_INTERNAL_INFORMATION)info)->IndexNumber.QuadPart =
                (LONGLONG)f->node + 1;
            ios->Information = sizeof(XBOX_FILE_INTERNAL_INFORMATION);
            break;
        case XboxFileNetworkOpenInformation:
            fill_net(n, (PXBOX_FILE_NETWORK_OPEN_INFORMATION)info);
            ios->Information = sizeof(XBOX_FILE_NETWORK_OPEN_INFORMATION);
            break;
        default:
            xbox_log(XBOX_LOG_WARN, XBOX_LOG_FILE,
                     "disc image NtQueryInformationFile: unhandled class %d", cls);
            *st = STATUS_NOT_IMPLEMENTED;
            goto out;
    }
    ios->Status = *st = STATUS_SUCCESS;
out:
    ReleaseSRWLockExclusive(&s_lock);
    return 1;
}

int xbox_disc_set_info(HANDLE h, PXBOX_IO_STATUS_BLOCK ios, PVOID info,
                       ULONG len, XBOX_FILE_INFORMATION_CLASS cls, NTSTATUS *st)
{
    disc_file *f;
    NTSTATUS r = STATUS_SUCCESS;

    (void)len;
    AcquireSRWLockExclusive(&s_lock);
    f = find_file(h, NULL);
    if (!f) {
        ReleaseSRWLockExclusive(&s_lock);
        return 0;
    }
    if (!ios || !info) {
        *st = STATUS_INVALID_PARAMETER;
        goto out;
    }
    switch (cls) {
        case XboxFilePositionInformation: {
            LONGLONG pos = ((PXBOX_FILE_POSITION_INFORMATION)info)->CurrentByteOffset.QuadPart;
            if (pos < 0)
                r = STATUS_INVALID_PARAMETER;
            else
                f->pos = (uint64_t)pos;
            break;
        }
        case XboxFileBasicInformation:
            break;              /* times and attributes: nothing to keep */
        case XboxFileEndOfFileInformation:
        case XboxFileAllocationInformation:
        case XboxFileDispositionInformation:
            r = STATUS_MEDIA_WRITE_PROTECTED;
            break;
        default:
            xbox_log(XBOX_LOG_WARN, XBOX_LOG_FILE,
                     "disc image NtSetInformationFile: unhandled class %d", cls);
            *st = STATUS_NOT_IMPLEMENTED;
            goto out;
    }
    ios->Status = *st = r;
    ios->Information = 0;
out:
    ReleaseSRWLockExclusive(&s_lock);
    return 1;
}

int xbox_disc_query_volume(HANDLE h, PXBOX_IO_STATUS_BLOCK ios, PVOID info,
                           ULONG len, XBOX_FS_INFORMATION_CLASS cls, NTSTATUS *st)
{
    AcquireSRWLockExclusive(&s_lock);
    if (!find_file(h, NULL)) {
        ReleaseSRWLockExclusive(&s_lock);
        return 0;
    }
    if (!ios || !info) {
        *st = STATUS_INVALID_PARAMETER;
    } else if (cls == XboxFileFsSizeInformation) {
        PXBOX_FILE_FS_SIZE_INFORMATION s = (PXBOX_FILE_FS_SIZE_INFORMATION)info;
        s->BytesPerSector = (ULONG)DISC_SECTOR;
        s->SectorsPerAllocationUnit = 1;
        s->TotalAllocationUnits.QuadPart =
            (LONGLONG)((s_image_size - s_base) / DISC_SECTOR);
        s->AvailableAllocationUnits.QuadPart = 0;
        ios->Status = *st = STATUS_SUCCESS;
        ios->Information = sizeof *s;
    } else if (cls == XboxFileFsVolumeInformation) {
        *st = xbox_FsVolumeInformation(ios, info, len, 0x44495343u /* "DISC" */, "");
    } else {
        xbox_log(XBOX_LOG_WARN, XBOX_LOG_FILE,
                 "disc image NtQueryVolumeInformationFile: unhandled class %d", cls);
        *st = STATUS_NOT_IMPLEMENTED;
    }
    ReleaseSRWLockExclusive(&s_lock);
    return 1;
}

int xbox_disc_flush(HANDLE h, PXBOX_IO_STATUS_BLOCK ios, NTSTATUS *st)
{
    int mine;
    AcquireSRWLockExclusive(&s_lock);
    mine = find_file(h, NULL) != NULL;
    ReleaseSRWLockExclusive(&s_lock);
    if (!mine)
        return 0;
    if (ios) {
        ios->Status = STATUS_SUCCESS;
        ios->Information = 0;
    }
    *st = STATUS_SUCCESS;
    return 1;
}

int xbox_disc_query_full(PXBOX_OBJECT_ATTRIBUTES oa,
                         PXBOX_FILE_NETWORK_OPEN_INFORMATION info, NTSTATUS *st)
{
    uint32_t node;
    if (!resolve(oa, &node))
        return 0;
    if (!info) {
        *st = STATUS_INVALID_PARAMETER;
        return 1;
    }
    AcquireSRWLockExclusive(&s_lock);
    fill_net(&s_nodes[node], info);
    ReleaseSRWLockExclusive(&s_lock);
    *st = STATUS_SUCCESS;
    return 1;
}

int xbox_disc_query_dir(HANDLE h, PXBOX_IO_STATUS_BLOCK ios, PVOID info, ULONG len,
                        XBOX_FILE_INFORMATION_CLASS cls, PXBOX_ANSI_STRING name,
                        BOOLEAN restart, NTSTATUS *st)
{
    disc_file *f;
    uint32_t i, hit = 0;
    int found = 0;

    AcquireSRWLockExclusive(&s_lock);
    f = find_file(h, NULL);
    if (!f) {
        ReleaseSRWLockExclusive(&s_lock);
        return 0;
    }
    if (!ios || !info) {
        *st = STATUS_INVALID_PARAMETER;
        goto out;
    }
    ios->Information = 0;
    if (cls != XboxFileDirectoryInformation) {
        ios->Status = *st = STATUS_INVALID_INFO_CLASS;
        goto out;
    }
    if (!s_nodes[f->node].is_dir) {
        ios->Status = *st = STATUS_INVALID_PARAMETER;
        goto out;
    }
    if (restart || !f->enum_started) {
        f->enum_started = 1;
        f->enum_done = 0;
        f->enum_next = 1;       /* [0] is the root, never a child */
        if (name && name->Buffer && name->Length > 0) {
            USHORT n = name->Length;
            if (n >= sizeof f->pattern)
                n = sizeof f->pattern - 1;
            memcpy(f->pattern, name->Buffer, n);
            f->pattern[n] = '\0';
        } else {
            strcpy(f->pattern, "*");
        }
        if (!strcmp(f->pattern, "*.*"))
            strcpy(f->pattern, "*");
    }
    if (!f->enum_done) {
        for (i = f->enum_next; i < s_node_count; i++) {
            if (s_nodes[i].parent == f->node && wild(f->pattern, leaf_of(s_nodes[i].path))) {
                hit = i;
                found = 1;
                break;
            }
        }
    }
    if (!found) {
        f->enum_done = 1;
        ios->Status = *st = STATUS_NO_MORE_FILES;
        goto out;
    }
    {
        PXBOX_FILE_DIRECTORY_INFORMATION e = (PXBOX_FILE_DIRECTORY_INFORMATION)info;
        XBOX_FILE_NETWORK_OPEN_INFORMATION net;
        const char *leaf = leaf_of(s_nodes[hit].path);
        ULONG header = (ULONG)((ULONG_PTR)&((PXBOX_FILE_DIRECTORY_INFORMATION)0)->FileName);
        ULONG name_len = (ULONG)strlen(leaf);

        if (len < header) {
            ios->Status = *st = STATUS_BUFFER_TOO_SMALL;
            goto out;
        }
        f->enum_next = hit + 1;
        memset(e, 0, len);
        fill_net(&s_nodes[hit], &net);
        e->CreationTime = net.CreationTime;
        e->LastAccessTime = net.LastAccessTime;
        e->LastWriteTime = net.LastWriteTime;
        e->ChangeTime = net.ChangeTime;
        e->EndOfFile = net.EndOfFile;
        e->AllocationSize = net.AllocationSize;
        e->FileAttributes = net.FileAttributes;
        e->FileNameLength = name_len;
        if (header + name_len <= len) {
            memcpy(e->FileName, leaf, name_len);
            ios->Status = *st = STATUS_SUCCESS;
            ios->Information = header + name_len;
        } else {
            memcpy(e->FileName, leaf, len - header);
            ios->Status = *st = STATUS_BUFFER_OVERFLOW;
            ios->Information = len;
        }
    }
out:
    ReleaseSRWLockExclusive(&s_lock);
    return 1;
}

int xbox_disc_rename(HANDLE h, NTSTATUS *st)
{
    int mine;
    AcquireSRWLockExclusive(&s_lock);
    mine = find_file(h, NULL) != NULL;
    ReleaseSRWLockExclusive(&s_lock);
    if (!mine)
        return 0;
    *st = STATUS_MEDIA_WRITE_PROTECTED;
    return 1;
}

int xbox_disc_duplicate(HANDLE src, HANDLE *dup, int close_source)
{
    disc_file *f;
    size_t slot;

    *dup = NULL;
    AcquireSRWLockExclusive(&s_lock);
    f = find_file(src, NULL);
    if (!f) {
        ReleaseSRWLockExclusive(&s_lock);
        return 0;
    }
    *dup = add_handle(f);
    if (close_source && find_file(src, &slot)) {
        s_handles[slot] = s_handles[--s_handle_count];
        release_file(f);
    } else {
        close_source = 0;
    }
    ReleaseSRWLockExclusive(&s_lock);
    if (close_source)
        CloseHandle(src);
    return 1;
}
