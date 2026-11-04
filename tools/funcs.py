import struct, sys
img = open(sys.argv[1],'rb').read(); hdr = int(sys.argv[2],16)
ver, ptr_enc, cnt_enc, tbl_enc = img[hdr:hdr+4]
assert (ver, cnt_enc, tbl_enc) == (1, 0x03, 0x3b), (ver,ptr_enc,cnt_enc,tbl_enc)
p = hdr+4
p += 4  # eh_frame_ptr (sdata4 pcrel)
n = struct.unpack_from('<I', img, p)[0]; p += 4
starts = []
for i in range(n):
    ip, fde = struct.unpack_from('<ii', img, p + i*8)
    starts.append(hdr + ip)
starts.sort()
for s in starts: print(hex(s))
print(n, file=sys.stderr)
