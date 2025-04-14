from argparse import ArgumentParser
from parser.process_dataset import sequences_and_residue_pairs
import tempfile
import os

def parse_arguments():
    parser = ArgumentParser(description='Convert mzIdentML to AlphaLink2 crosslinking format')
    parser.add_argument('--mzidentml',
                        help='mzIdentML input file',
                        required=True)
    parser.add_argument('--fdr',
                        help='False discovery rate',
                        default=0.2,
                        type=float,
                        required=False)
    args = parser.parse_args()
    return args

# Complete sequence of chain IDs supported by the PDB format
chains = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

def main():
    args = parse_arguments()

    fname = os.path.splitext(os.path.basename(args.mzidentml))[0]

    temp_dir = tempfile.TemporaryDirectory()
    result = sequences_and_residue_pairs(args.mzidentml, temp_dir.name)
    temp_dir.cleanup()

    sequence_chain_mapping = {}
    with open(fname+'.fasta', 'w') as f:
        for sequence, chain in zip(result['sequences'], chains):
            f.write(f'>{chain}|{sequence['id']}\n')
            f.write(f'{sequence['sequence']}\n')
            sequence_chain_mapping[sequence['id']] = chain


    with open(fname+'.txt','w') as f:
        for link in result['residue_pairs']:
            if not link['prot1'] in sequence_chain_mapping:
                continue
            if not link['prot2'] in sequence_chain_mapping:
                continue
            f.write(f'{link['pos1']} {sequence_chain_mapping[link['prot1']]} {link['pos2']} {sequence_chain_mapping[link['prot2']]} {args.fdr}\n')

if __name__ == "__main__":
    main()