#!/usr/bin/env bash
#
# Restore a Raspberry Pi SD card from a partclone backup.
#
# Assumes the target card may be in ANY state (blank, wrong partitions,
# random data): it force-rewrites the partition table and restores both
# filesystems from scratch. Everything currently on the card is DESTROYED.
#
# Reads (from this script's directory):
#   - partition-table.sfdisk : partition layout + disk identifier
#   - boot.img.<zst|gz>      : the vfat boot partition image
#   - root.img.<zst|gz>      : the ext4 root partition image
#
# NOTE: partclone restores each filesystem at its original size (no resize),
# so the target card must be at least as large as the one backed up.
#
# Usage:
#   sudo ./restore.sh DEVICE
#   sudo ./restore.sh /dev/sda
#
set -euo pipefail

DEVICE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TABLE_FILE="$SCRIPT_DIR/partition-table.sfdisk"

# --- helpers ---------------------------------------------------------------

die() { echo "ERROR: $*" >&2; exit 1; }

part() {
    local dev="$1" idx="$2"
    if [[ "$dev" =~ [0-9]$ ]]; then echo "${dev}p${idx}"; else echo "${dev}${idx}"; fi
}

partclone_for() {
    case "$1" in
        ext2|ext3|ext4) echo partclone.extfs ;;
        vfat|fat|fat16|fat32) echo partclone.vfat ;;
        *) die "Unsupported filesystem type '$1'." ;;
    esac
}

# Decompressor for an image file, based on its extension.
decompress_for() {
    case "$1" in
        *.zst) echo "zstd -dc" ;;
        *.gz)  echo "gzip -dc" ;;
        *)     echo "cat" ;;
    esac
}

# Find the single image matching a prefix (boot/root), whatever its extension.
find_image() {
    local prefix="$1" matches
    matches=("$SCRIPT_DIR/$prefix".img.*)
    [[ -f "${matches[0]}" ]] || die "No image found for '$prefix' (expected $prefix.img.zst/gz)."
    [[ ${#matches[@]} -eq 1 ]] || die "Multiple images for '$prefix': ${matches[*]}"
    echo "${matches[0]}"
}

# Restore one compressed image onto a partition with the right partclone tool.
restore_part() {
    local img="$1" dev="$2" fstype tool decomp
    # Derive fstype from the source partition table's filesystem in the image?
    # partclone stores it, but we simply map by the same rule used at backup:
    # boot=vfat, root=extfs is inferred from the image filename prefix.
    case "$img" in
        *boot.img.*) tool=partclone.vfat ;;
        *root.img.*) tool=partclone.extfs ;;
        *) die "Cannot determine partclone tool for $img" ;;
    esac
    decomp="$(decompress_for "$img")"
    echo "==> Restoring $img -> $dev with $tool"
    $decomp "$img" | "$tool" -r -s - -o "$dev"
}

# --- checks ----------------------------------------------------------------

[[ $EUID -eq 0 ]] || die "Must be run as root (use sudo)."
command -v partclone.extfs >/dev/null || die "partclone is not installed (apt install partclone)."
[[ -n "$DEVICE" ]] || die "Usage: sudo ./restore.sh DEVICE (e.g. /dev/sda)"
[[ -b "$DEVICE" ]] || die "$DEVICE is not a block device."
[[ -f "$TABLE_FILE" ]] || die "Missing $TABLE_FILE (run backup.sh first)."

BOOT_IMG="$(find_image boot)"
ROOT_IMG="$(find_image root)"
BOOT_PART="$(part "$DEVICE" 1)"
ROOT_PART="$(part "$DEVICE" 2)"

# --- confirmation ----------------------------------------------------------

echo "==> Target device: $DEVICE"
lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINT "$DEVICE"
echo
echo "*** ALL DATA ON $DEVICE WILL BE ERASED ***"
read -r -p "Type the device path again to confirm ($DEVICE): " CONFIRM
[[ "$CONFIRM" == "$DEVICE" ]] || die "Confirmation did not match. Aborting."

# --- unmount anything on the target ----------------------------------------

echo "==> Unmounting any partitions of $DEVICE..."
while read -r p; do
    [[ -n "$p" ]] || continue
    while mp="$(findmnt -n -o TARGET --source "$p" 2>/dev/null | head -1)"; [[ -n "$mp" ]]; do
        echo "    umount $p ($mp)"
        umount "$p"
    done
done < <(lsblk -ln -o PATH "$DEVICE" | tail -n +2)

# --- wipe and rewrite the partition table ----------------------------------

echo "==> Wiping existing signatures on $DEVICE..."
wipefs -a "$DEVICE" >/dev/null

echo "==> Restoring partition table..."
sfdisk --wipe always --wipe-partitions always "$DEVICE" < "$TABLE_FILE"

echo "==> Re-reading partition table..."
partprobe "$DEVICE" 2>/dev/null || blockdev --rereadpt "$DEVICE"
udevadm settle 2>/dev/null || true
sleep 1

[[ -b "$BOOT_PART" ]] || die "Boot partition $BOOT_PART did not appear after repartitioning."
[[ -b "$ROOT_PART" ]] || die "Root partition $ROOT_PART did not appear after repartitioning."

# --- restore filesystems ---------------------------------------------------

restore_part "$BOOT_IMG" "$BOOT_PART"
restore_part "$ROOT_IMG" "$ROOT_PART"

sync
echo
echo "==> Restore complete. You can now eject $DEVICE."
lsblk -o NAME,SIZE,FSTYPE,LABEL "$DEVICE"
