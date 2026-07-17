#!/usr/bin/env bash
#
# Backup a Raspberry Pi SD card using partclone.
#
# Unlike a raw `dd`, partclone reads only the *used* blocks of each
# filesystem (via the fs bitmap), so a mostly-empty 60 GB card produces
# an image of only a couple of GB in a couple of minutes.
#
# partclone is used here instead of fsarchiver because it opens the
# filesystem read-only and tolerates the newer ext4 features (orphan_file)
# that this host's fsarchiver 0.8.6 refuses.
#
# Saved into this script's directory:
#   - partition-table.sfdisk : partition layout + disk identifier
#                              (preserves PARTUUIDs used by cmdline.txt/fstab)
#   - boot.img.zst           : the vfat boot partition (only used blocks)
#   - root.img.zst           : the ext4 root partition (only used blocks)
#
# Usage:
#   sudo ./backup.sh [DEVICE]
#   sudo ./backup.sh /dev/sda        # explicit device (default)
#
set -euo pipefail

DEVICE="${1:-/dev/sda}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TABLE_FILE="$SCRIPT_DIR/partition-table.sfdisk"

# --- helpers ---------------------------------------------------------------

die() { echo "ERROR: $*" >&2; exit 1; }

# Partition node for a device+index, handling both sdaN and mmcblk0pN styles.
part() {
    local dev="$1" idx="$2"
    if [[ "$dev" =~ [0-9]$ ]]; then echo "${dev}p${idx}"; else echo "${dev}${idx}"; fi
}

# Pick the right partclone binary for a filesystem type.
partclone_for() {
    case "$1" in
        ext2|ext3|ext4) echo partclone.extfs ;;
        vfat|fat|fat16|fat32) echo partclone.vfat ;;
        *) die "Unsupported filesystem type '$1'." ;;
    esac
}

# Compress command: prefer zstd (fast, multi-threaded), else gzip.
if command -v zstd >/dev/null; then COMPRESS=(zstd -T0 -3); EXT=zst
else COMPRESS=(gzip -c); EXT=gz; fi

# Clone one partition to a compressed image.
clone_part() {
    local dev="$1" out="$2" fstype tool
    fstype="$(blkid -o value -s TYPE "$dev")" || die "Cannot read fstype of $dev."
    tool="$(partclone_for "$fstype")"
    echo "==> Cloning $dev ($fstype) with $tool -> $out"
    "$tool" -c -s "$dev" -o - | "${COMPRESS[@]}" > "$out"
}

# --- checks ----------------------------------------------------------------

[[ $EUID -eq 0 ]] || die "Must be run as root (use sudo)."
command -v partclone.extfs >/dev/null || die "partclone is not installed (apt install partclone)."
[[ -b "$DEVICE" ]] || die "$DEVICE is not a block device."

BOOT_PART="$(part "$DEVICE" 1)"
ROOT_PART="$(part "$DEVICE" 2)"
[[ -b "$BOOT_PART" ]] || die "Boot partition $BOOT_PART not found."
[[ -b "$ROOT_PART" ]] || die "Root partition $ROOT_PART not found."

echo "==> Source device: $DEVICE"
lsblk -o NAME,SIZE,FSTYPE,LABEL "$DEVICE"
echo

# --- unmount so the snapshot is consistent ---------------------------------

echo "==> Unmounting partitions of $DEVICE (if mounted)..."
for p in "$BOOT_PART" "$ROOT_PART"; do
    while mp="$(findmnt -n -o TARGET --source "$p" 2>/dev/null | head -1)"; [[ -n "$mp" ]]; do
        echo "    umount $p ($mp)"
        umount "$p"
    done
done

# --- save partition table --------------------------------------------------

echo "==> Saving partition table -> $TABLE_FILE"
sfdisk -d "$DEVICE" > "$TABLE_FILE"

# --- save filesystems ------------------------------------------------------

clone_part "$BOOT_PART" "$SCRIPT_DIR/boot.img.$EXT"
clone_part "$ROOT_PART" "$SCRIPT_DIR/root.img.$EXT"

sync
echo
echo "==> Backup complete."
ls -lh "$TABLE_FILE" "$SCRIPT_DIR"/boot.img.* "$SCRIPT_DIR"/root.img.*
