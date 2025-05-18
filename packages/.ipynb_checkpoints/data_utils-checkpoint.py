# ---- Helper Functions ----
import numpy as np
import gffutils

# ---- One-hot encoding dictionary ----
BASE_DICT = {'A': [1, 0, 0, 0],
             'C': [0, 1, 0, 0],
             'G': [0, 0, 1, 0],
             'T': [0, 0, 0, 1],
             'N': [0, 0, 0, 0]}

def one_hot_encode(seq):
    return np.array([BASE_DICT.get(base.upper(), [0, 0, 0, 0]) for base in seq]).T

def decode_one_hot(array):
    index_to_base = {tuple([1, 0, 0, 0]): 'A',
                     tuple([0, 1, 0, 0]): 'C',
                     tuple([0, 0, 1, 0]): 'G',
                     tuple([0, 0, 0, 1]): 'T',
                     tuple([0, 0, 0, 0]): 'N'}  # unknown or padding

    return ''.join(index_to_base.get(tuple(vec), 'N') for vec in array.T)

def parse_coords(coord_str):
    # format: "start-end"
    start, end = map(int, coord_str.split('-'))
    return start, end

def make_blocks(start, end, chrom, genome, PADDING=5000, BLOCK_SIZE = 15000):
    region_start = max(0, start - PADDING)
    region_end = end + PADDING
    sequence = genome[chrom][region_start:region_end].seq.upper()
    
    blocks = []
    for i in range(0, len(sequence) - BLOCK_SIZE + 1, PADDING):
        block_seq = sequence[i:i + BLOCK_SIZE]
        blocks.append((region_start + i, region_start + i + BLOCK_SIZE, block_seq))
    return blocks

def assign_labels(block_start, block_end, psi_dict, PADDING=5000, BLOCK_SIZE = 15000):
    labels = np.zeros((BLOCK_SIZE, 3))  # 12 for 1 tissue (spliced/unspliced/usage) x 4 tissues
    mid_start = PADDING
    mid_end = BLOCK_SIZE - PADDING

    for pos, psi in psi_dict.items():
        if block_start <= pos < block_end:
            rel_pos = pos - block_start
            if mid_start <= rel_pos < mid_end:
                # For this example: tissue is heart → positions 0–2
                labels[rel_pos, 0] = 1 if psi < 0.1 else 0
                labels[rel_pos, 1] = 0 if psi < 0.1 else 1
                labels[rel_pos, 2] = 0 if psi < 0.1 else psi
    #return labels[mid_start:mid_end]  # Return only the middle 5000
    return labels.T  # Return Full

def get_gene_bounds(gene_id, db_path='gtf.db'):
    db = gffutils.FeatureDB(db_path)
    try:
        gene = db[gene_id]
        chrom = gene.chrom
        strand = gene.strand
        tss = gene.start if strand == '+' else gene.end
        tes = gene.end if strand == '+' else gene.start
        return chrom, strand, tss, tes
    except Exception:
        return None