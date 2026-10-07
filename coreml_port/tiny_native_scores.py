"""Opt-in source-CPU score arithmetic for logical prefixes of 1–15 tokens.

The four nonsequential FP32 sum trees and their Q/row/key-column placement
were recovered from the installed Accelerate SGEMM through cancellation probes.
The frozen placement passed independent BF16 inputs for every score at square
Q=K=1..15. It is an empirical host/kernel-specific result, not a portable
identity for arbitrary FP32 operands or another BLAS build.

The caller applies the ordinary causal/key mask. Only the first 16 keys are
material for these prefixes; score slots 16..2047 are returned as zeros and
must be masked before softmax.
"""
import torch

from coreml_port.dynamic_queries import scores as ordinary_scores

# Post-recovery binary FP32 addition trees. Leaf i denotes q[..., i]*k[..., i].
# These literal constants are frozen from evidence/04f-tiny, with no runtime
# dependency on diagnostic scripts or evidence files.
TREE_A = [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[0,4],8],12],16],20],24],28],32],36],40],44],48],52],56],60],64],68],72],76],80],84],88],92],96],100],104],108],112],116],120],124],[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[1,5],9],13],17],21],25],29],33],37],41],45],49],53],57],61],65],69],73],77],81],85],89],93],97],101],105],109],113],117],121],125]],[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[2,6],10],14],18],22],26],30],34],38],42],46],50],54],58],62],66],70],74],78],82],86],90],94],98],102],106],110],114],118],122],126],[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[3,7],11],15],19],23],27],31],35],39],43],47],51],55],59],63],67],71],75],79],83],87],91],95],99],103],107],111],115],119],123],127]]]
TREE_F = [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[0,4],8],12],16],20],24],28],32],36],40],44],48],52],56],60],64],68],72],76],80],84],88],92],96],100],104],108],112],116],120],124],[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[2,6],10],14],18],22],26],30],34],38],42],46],50],54],58],62],66],70],74],78],82],86],90],94],98],102],106],110],114],118],122],126]],[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[1,5],9],13],17],21],25],29],33],37],41],45],49],53],57],61],65],69],73],77],81],85],89],93],97],101],105],109],113],117],121],125],[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[3,7],11],15],19],23],27],31],35],39],43],47],51],55],59],63],67],71],75],79],83],87],91],95],99],103],107],111],115],119],123],127]]]
TREE_T = [[[[[[[[[0,12],28],44],[[4,20],36]],52],[[[[8,24],40],[[[16,60],32],48]],56]],[[[[[[2,14],30],46],[[6,22],38]],54],[[[[10,26],42],[[[18,62],34],50]],58]]],[[[[[[[1,13],29],45],[[5,21],37]],53],[[[[9,25],41],[[[17,61],33],49]],57]],[[[[[[3,15],31],47],[[7,23],39]],55],[[[[11,27],43],[[[19,63],35],51]],59]]]],[[[[[[[[64,76],92],108],[[68,84],100]],116],[[[[72,88],104],[[[80,124],96],112]],120]],[[[[[[66,78],94],110],[[70,86],102]],118],[[[[74,90],106],[[[82,126],98],114]],122]]],[[[[[[[65,77],93],109],[[69,85],101]],117],[[[[73,89],105],[[[81,125],97],113]],121]],[[[[[[67,79],95],111],[[71,87],103]],119],[[[[75,91],107],[[[83,127],99],115]],123]]]]]
TREE_E = [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[0,2],4],6],8],10],12],14],16],18],20],22],24],26],28],30],32],34],36],38],40],42],44],46],48],50],52],54],56],58],60],62],[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[1,3],5],7],9],11],13],15],17],19],21],23],25],27],29],31],33],35],37],39],41],43],45],47],49],51],53],55],57],59],61],63]],64],66],68],70],72],74],76],78],80],82],84],86],88],90],92],94],96],98],100],102],104],106],108],110],112],114],116],118],120],122],124],126],[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[65,67],69],71],73],75],77],79],81],83],85],87],89],91],93],95],97],99],101],103],105],107],109],111],113],115],117],119],121],123],125],127]]


def _mode_table():
    # Codes: 0=sequential, 1=four-lane adjacent, 2=four-lane shuffle,
    # 3=Q6/7 tail tree, 4=Q8 tile tree. Entries outside logical Q are inert.
    modes = [[[0 for _ in range(16)] for _ in range(32)] for _ in range(16)]
    modes[3][2][:2] = [1, 1]
    modes[3][2][2] = 2
    for r in range(4):
        modes[4][r][:4] = [1] * 4
    for r in range(4):
        modes[5][r][:4] = [1] * 4
    modes[5][4][:4] = [1] * 4
    modes[5][4][4] = 2
    for size in (6, 7):
        for r in range(4, size):
            modes[size][r][:size] = [3] * size
    for r in range(8):
        modes[8][r][:8] = [4] * 8
    for size in (9, 10, 11):
        for r in range(8):
            modes[size][r][:8] = [4] * 8
        for r in range(8, size):
            modes[size][r][8:size] = [3] * (size - 8)
    for r in range(12):
        modes[12][r][:8] = [4] * 8
    for size in (13, 14):
        for r in range(12):
            modes[size][r][:8] = [4] * 8
        for r in range(12, size):
            modes[size][r][8:size] = [3] * (size - 8)
    return torch.tensor(modes, dtype=torch.int32)


MODE_TABLE = _mode_table()


def _reduce(tree, products):
    if isinstance(tree, int):
        return products[tree]
    return _reduce(tree[0], products) + _reduce(tree[1], products)


def _short_raw(query, expanded_keys):
    """Return unscaled [1,10,Q,16] dots from the measured short geometry."""
    keys = expanded_keys[:, :16]
    products = [query[..., i].unsqueeze(2) * keys[..., i].unsqueeze(1)
                for i in range(128)]
    sequential = products[0]
    for product in products[1:]:
        sequential = sequential + product
    reduced = [sequential, _reduce(TREE_A, products), _reduce(TREE_F, products),
               _reduce(TREE_T, products), _reduce(TREE_E, products)]
    # Each reduction is [1,Q,16,10]; output scores are [1,10,Q,16].
    reduced = [value.permute(0, 3, 1, 2) for value in reduced]
    rows = torch._shape_as_tensor(query)[1]
    safe_rows = torch.clamp(rows, min=1, max=15)
    modes = torch.index_select(MODE_TABLE, 0, safe_rows.reshape(1))[0][:rows, :16]
    active = reduced[0]
    for mode in range(1, 5):
        active = torch.where(modes[None, None] == mode, reduced[mode], active)
    return active


def scores(query, expanded_keys):
    """Return [1,10,Q,2048] scores; Q>=16 uses the ordinary score expression."""
    rows = torch._shape_as_tensor(query)[1]
    active = _short_raw(query, expanded_keys) * .1
    inactive = torch.zeros_like(active[..., :1]).expand(-1, -1, -1, 2032)
    short_scores = torch.cat((active, inactive), dim=-1)
    full_scores = ordinary_scores(query, expanded_keys)
    return torch.where(rows < 16, short_scores, full_scores)
