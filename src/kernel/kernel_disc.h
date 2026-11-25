/*
 * kernel_disc.h - the game's files read straight from its disc image
 *
 * Off until a host opens an image (xbox_DiscImageOpen) or RECOMP_DISC_IMAGE
 * names one. With it on, a guest open of a path in the game files tree
 * (kernel_path.c's xbox_path_game_rest: D:, CdRom0, and the E:, C:, Y: roots
 * the game dir stands in for) is served from the image when the name is on
 * the disc, read-only, as a DVD is. Anything else, including a name that is
 * not on the disc and a write anywhere outside D: and CdRom0, goes on to the
 * disk backend at <game_dir>, so a title that writes beside its files still
 * can.
 *
 * The image is an XDVDFS game partition, bare (extract-xiso, xdvdfs) or at
 * one of the offsets a full dump puts it (tools/xiso/xdvdfs.py lists them).
 * It is read once to build the name table and then on demand; nothing is
 * copied out. RECOMP_DISC_TRACE=1 logs each name it serves or passes on.
 *
 * kernel_file.c's xbox_Nt*File functions ask this store after the RAM cache
 * and before the disk. Each xbox_disc_* call returns 1 when the path or
 * handle is the image's (the NTSTATUS in *st) and 0 to leave it to the disk.
 */
#ifndef KERNEL_DISC_H
#define KERNEL_DISC_H

#include "kernel.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opens an image and turns the store on: 1, or 0 with the reason in err.
 * Before the guest starts; a second call is refused. */
int xbox_DiscImageOpen(const char *path, char *err, size_t err_n);
/* The open image's path, or NULL. */
const char *xbox_DiscImagePath(void);
/* A whole file from the image by its path below the root ("default.xbe",
 * "media\\intro.xmv"), into a malloc'd buffer: 1, or 0 when it is not there.
 * For the host's XBE load, which happens before any guest open. */
int xbox_DiscReadFile(const char *rel, void **data, size_t *size);

int xbox_disc_active(void);

int xbox_disc_create(PHANDLE out, ACCESS_MASK access, PXBOX_OBJECT_ATTRIBUTES oa,
                     PXBOX_IO_STATUS_BLOCK ios, ULONG disposition, ULONG options,
                     NTSTATUS *st);
int xbox_disc_read(HANDLE h, HANDLE event, PXBOX_IO_STATUS_BLOCK ios,
                   PVOID buf, ULONG len, PLARGE_INTEGER off, NTSTATUS *st);
int xbox_disc_write(HANDLE h, PXBOX_IO_STATUS_BLOCK ios, NTSTATUS *st);
int xbox_disc_close(HANDLE h);
int xbox_disc_delete(PXBOX_OBJECT_ATTRIBUTES oa, NTSTATUS *st);
int xbox_disc_query_info(HANDLE h, PXBOX_IO_STATUS_BLOCK ios, PVOID info,
                         ULONG len, XBOX_FILE_INFORMATION_CLASS cls, NTSTATUS *st);
int xbox_disc_set_info(HANDLE h, PXBOX_IO_STATUS_BLOCK ios, PVOID info,
                       ULONG len, XBOX_FILE_INFORMATION_CLASS cls, NTSTATUS *st);
int xbox_disc_query_volume(HANDLE h, PXBOX_IO_STATUS_BLOCK ios, PVOID info,
                           ULONG len, XBOX_FS_INFORMATION_CLASS cls, NTSTATUS *st);
int xbox_disc_flush(HANDLE h, PXBOX_IO_STATUS_BLOCK ios, NTSTATUS *st);
int xbox_disc_query_full(PXBOX_OBJECT_ATTRIBUTES oa,
                         PXBOX_FILE_NETWORK_OPEN_INFORMATION info, NTSTATUS *st);
int xbox_disc_query_dir(HANDLE h, PXBOX_IO_STATUS_BLOCK ios, PVOID info, ULONG len,
                        XBOX_FILE_INFORMATION_CLASS cls, PXBOX_ANSI_STRING name,
                        BOOLEAN restart, NTSTATUS *st);
int xbox_disc_rename(HANDLE h, NTSTATUS *st);
/* NtDuplicateObject: a new handle on the same file object (shared position).
 * Returns 1 for an image handle, with *dup NULL when it failed. */
int xbox_disc_duplicate(HANDLE src, HANDLE *dup, int close_source);

#ifdef __cplusplus
}
#endif

#endif /* KERNEL_DISC_H */
