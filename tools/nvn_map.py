#!/usr/bin/env python3
"""Map NVN proc names -> global pointer slots (via GOT) -> call sites in main. Writes nvn_map.tsv."""
import re, sys, collections
sys.path.insert(0, __file__.rsplit('/',1)[0]); import nso
n = nso.load(sys.argv[1] if len(sys.argv) > 1 else 'exefs/main'); img = n['image']; dyn = nso.parse_dyn(img)
rel = {off: add for off, t, s, add in dyn['rela'] if t == 0x403}  # RELATIVE: GOT entry -> slot address
names = {m.start(1): m.group(1).decode() for m in re.finditer(rb'(?<=\x00)(nvn[A-Z][A-Za-z0-9]+)(?=\x00)', img)}
insn = re.compile(r'^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$')
seq = []
for l in open('main_addr.asm'):
    m = insn.match(l)
    if m: seq.append((int(m.group(1), 16), m.group(2), m.group(3)))
def adrp_target(args): return int(args.split(',')[1].strip().split()[0], 16)
slot_of = {}
for i, (a, op, args) in enumerate(seq):
    if op != 'add': continue
    p = [x.strip() for x in args.split(',')]
    if len(p) < 3 or not p[2].startswith('#'): continue
    # find adrp for p[1] just before
    for k in range(i - 1, i - 4, -1):
        if seq[k][1] == 'adrp' and seq[k][2].split(',')[0].strip() == p[1]:
            v = adrp_target(seq[k][2]) + int(p[2][1:].split()[0], 0)
            if v in names:
                # next: blr, then adrp x8 / ldr x8,[x8,#o] / str x0,[x8]
                pages = {}
                for j in range(i, i + 14):
                    _, op2, a2 = seq[j]
                    if op2 == 'adrp': pages[a2.split(',')[0].strip()] = adrp_target(a2)
                    mm = re.match(r'(x\d+), \[(x\d+), #(0x[0-9a-f]+|\d+)\]', a2)
                    if op2 == 'ldr' and mm and mm.group(2) in pages and j > i:
                        got = pages[mm.group(2)] + int(mm.group(3), 0)
                        if got in rel and seq[j + 1][1] in ('adrp', 'str', 'add', 'mov') or got in rel:
                            slot_of[names[v]] = (got, rel.get(got))
                            break
            break
print(len(slot_of), 'of', len(names), 'procs mapped to GOT entries', file=sys.stderr)
# call sites: adrp X, page; ldr X, [X, #o] (got) ; ldr Y, [X] ; ... blr Y
got_to_name = {g: nm for nm, (g, s) in slot_of.items()}
calls = collections.defaultdict(list)
for i, (a, op, args) in enumerate(seq):
    if op != 'ldr': continue
    mm = re.match(r'(x\d+), \[(x\d+), #(0x[0-9a-f]+|\d+)\]', args)
    if not mm: continue
    for k in range(i - 1, i - 6, -1):
        if seq[k][1] == 'adrp' and seq[k][2].split(',')[0].strip() == mm.group(2):
            got = adrp_target(seq[k][2]) + int(mm.group(3), 0)
            nm = got_to_name.get(got)
            if nm:
                xr = mm.group(1)
                for j in range(i + 1, i + 30):
                    _, op2, a2 = seq[j]
                    if op2 == 'ldr' and re.match(r'(x\d+), \[%s\]$' % xr, a2):
                        fr = a2.split(',')[0]
                        for q in range(j + 1, j + 40):
                            if seq[q][1] == 'blr' and seq[q][2].strip() == fr:
                                calls[nm].append(seq[q][0]); break
                            if seq[q][1] in ('ret', 'b') : break
                        break
            break
with open('nvn_map.tsv', 'w') as f:
    for nm in sorted(slot_of):
        g, s = slot_of[nm]
        f.write(f"{nm}\t{g:x}\t{(s or 0):x}\t{len(calls[nm])}\t{' '.join('%x' % c for c in calls[nm][:6])}\n")
