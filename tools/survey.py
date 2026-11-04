import sys, struct, re, collections
sys.path.insert(0,'tools'); import nso
n = nso.load('/mnt/user-data/uploads/DarkSoulsR/exefs/main'); img = n['image']; d = nso.parse_dyn(img)
names = {0x401:'RELATIVE',0x402:'GLOB_DAT',0x403:'JUMP_SLOT',0x101:'ABS64',0x406:'TLS_TPREL?',0x407:'TLSDESC'}
print('relocs:', collections.Counter(names.get(t,hex(t)) for _,t,_,_ in d['rela']))
print('DT_PLTREL size / JMPREL:', d['tags'].get(2), d['tags'].get(23))
# PLT stub detection: adrp x16; ldr x17,[x16,#]; add x16; br x17
text = img[:n['segs'][0]['size']]
plt = []
for o in range(0, len(text)-16, 4):
    w0, w1, w2, w3 = struct.unpack_from('<IIII', text, o)
    if (w0 & 0x9f00001f)==0x90000010 and (w1 & 0xffc003ff)==0xf9400211 and w3==0xd61f0220:
        plt.append(o)
print('PLT stubs:', len(plt), hex(plt[0]) if plt else None, hex(plt[-1]) if plt else None)
# RTTI: mangled typeinfo name strings in rodata
ro = img[n['segs'][1]['moff']:n['segs'][1]['moff']+n['segs'][1]['size']]
cls = set(m.group(0).decode() for m in re.finditer(rb'(?<=\0)N?[0-9]+[A-Za-z_][A-Za-z0-9_]{2,}(?:[0-9]+[A-Za-z_][A-Za-z0-9_]*)*E?(?=\0)', ro))
cls = {c for c in cls if re.fullmatch(r'(N(\d+\w+?)+E|\d+\w+)', c)}
print('RTTI type-name strings:', len(cls))
ns = collections.Counter()
for c in cls:
    m = re.match(r'N?(\d+)', c); L=int(m.group(1)); ns[c[m.end():m.end()+L]] += 1
print('top namespaces/classes:', ns.most_common(25))
open('work/main.rtti.txt','w').write('\n'.join(sorted(cls)))
