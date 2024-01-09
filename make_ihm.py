# Based on https://github.com/grandrea/alphalink-ihm-template/

import ihm
import ihm.location
import ihm.dataset
import ihm.restraint
import ihm.protocol
import ihm.model
import ihm.cross_linkers
import ihm.reference
import ihm.dumper
import ihm.representation
import pandas as pd
import ihm.reader
import ihm.citations
import sys
from Bio import SeqIO, pairwise2
import json
import pickle
import gzip

import requests as r
from io import StringIO

from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser(description='Generate distograms for binary contacts')
    parser.add_argument('--cif',
                        help='Model file',
                        required=True)
    parser.add_argument('--chain_id',
                        help='Chain id json mapping',
                        required=True)
    parser.add_argument('--db_code',
                        help='Database accession code',
                        required=True)
    parser.add_argument('--crosslinks',
                        help='Crosslink pickle',
                        required=True)
    parser.add_argument('--crosslinker',
                        help='Crosslinker',
                        choices=['dsso','sda','photoAA'])
    parser.add_argument('--protein_name',
                        help='Protein name',
                        required=True)
    parser.add_argument('--distance',
                        help='Distance of crosslinker',
                        default=25,
                        type=int,
                        required=False)  
    parser.add_argument('--output',
                        help='Output CSV with distogram restraints',
                        required=True)                   
    args = parser.parse_args()
    return args


def get_uniprot_fasta(uniprot_id):
    response = r.post('http://www.uniprot.org/uniprot/'+uniprot_id+'.fasta')
    data = ''.join(response.text)
    seq = list(SeqIO.parse(StringIO(data), 'fasta'))
    accession_code = seq[0].id.split('|')[1]
    return accession_code, str(seq[0].seq)


def longestSubstringFinder(string1, string2):
    answer = ""
    len1, len2 = len(string1), len(string2)
    for i in range(len1):
        for j in range(len2):
            lcs_temp = 0
            match = ''
            while ((i+lcs_temp < len1) and (j+lcs_temp<len2) and string1[i+lcs_temp] == string2[j+lcs_temp]):
                match += string2[j+lcs_temp]
                lcs_temp += 1
            if len(match) > len(answer):
                answer = match
    return answer

def get_alignment_indices(db_seq, seq):
    res = longestSubstringFinder(db_seq, seq)
    db_seq_begin = db_seq.find(res)
    db_seq_end = db_seq_begin + len(res)
    seq_begin = seq.find(res)
    seq_end = seq_begin + len(res)

    return db_seq_begin + 1, db_seq_end, seq_begin + 1, seq_end

def main():
    args = parse_arguments()

    with open(args.cif) as fh:
        system = ihm.reader.read(fh)
    system=system[0]

    chain_id = json.load(open(args.chain_id,'r'))

    mapping = {}
    entity_mapping = {}
    asym_mapping = {}

    for chain, values in chain_id.items():
        mapping[values['sequence']] = (chain, values['description'])

    for i, entity in enumerate(system.entities):
        entity_sequence = ''.join([s.code for s in entity.sequence])
        chain, description = mapping[entity_sequence]
        entity_mapping[chain] = entity
        entity.description = description
        uniprot_accession_code, uniprot_sequence = get_uniprot_fasta(description)
        model_sequence = ihm.reference.Sequence('UNP', chain, uniprot_accession_code, uniprot_sequence)
        db_begin,db_end,seq_begin,seq_end = get_alignment_indices(uniprot_sequence, entity_sequence)
        alignment = ihm.reference.Alignment(db_begin=db_begin,db_end=db_end,entity_begin=seq_begin,entity_end=seq_end)
        model_sequence.alignments.append(alignment)
        entity.references = [model_sequence]

        asym = system.asym_units[i]
        asym_mapping[chain] = asym
        asym.details = description


    system.title = "Integrative model of %s by crosslinking MS and deep learning" % (args.protein_name)

    #change authors as needed
    system.authors = ["Kolja Stahl",
                    "Oliver Brock",
                    "Juri Rappsilber"]

    citation = ihm.Citation(
        pmid=None,
        title='Modelling protein complexes with crosslinking mass spectrometry and deep learning',
        journal='biorxiv', volume=None, page_range=(1, 2), year=2023,
        authors=system.authors,
        doi="https://doi.org/10.1101/2023.06.07.544059")
    
    system.citations.append(citation)

    alphalink_software = ihm.Software(name="AlphaLink2",
                                    classification="model building",
                                    description="Modelling protein complexes with crosslinking mass spectrometry and deep learning",
                                    location="https://github.com/Rappsilber-Laboratory/AlphaLink2",
                                    version="1.0",
                                    citation=citation)

    system.software.append(alphalink_software)


    #define crosslinker. Here, photo-leucine.
    # photo_leucine = ihm.ChemDescriptor(auth_name="L-Photo-Leucine",
    #                                    chem_comp_id=None,
    #                                    smiles="CC1(C[C@H](N)C(O)=O)N=N1",
    #                                    inchi="1S/C5H9N3O2/c1-5(7-8-5)2-3(6)4(9)10/h3H,2,6H2,1H3,(H,9,10)/t3-/m0/s1",
    #                                    inchi_key="MJRDGTVDJKACQZ-VKHMYHEASA-N",
    #                                    common_name="L-Photo-Leucine")


    #change to PRIDE or whatever repository you have your cx-ms results in
    # crosslink_dataset = ihm.dataset.CXMSDataset(ihm.location.DatabaseLocation(db_name="jPOSTrepo",
    #                                             db_code="JPST001851"))

    # PXD020453
    crosslink_dataset = ihm.dataset.CXMSDataset(ihm.location.PRIDELocation(db_code=args.db_code))

    #if not using photo-leucine, use ihm.crosslinkers definitions
    crosslink_restraint = ihm.restraint.CrossLinkRestraint(dataset=crosslink_dataset,
                                                        linker=ihm.cross_linkers.sda)

    # Usually cross-links use an upper bound restraint on the distance
    distance = ihm.restraint.UpperBoundDistanceRestraint(args.distance)

    crosslinks = pickle.load(gzip.open(args.crosslinks,'rb'))

    for chain1, v in crosslinks.items():
        entity1 = entity_mapping[chain1]
        asym1 = asym_mapping[chain1]
        for chain2, links in v.items():
            entity2 = entity_mapping[chain2]
            asym2 = asym_mapping[chain2]

            for i,j,fdr in links:
                # This assumes that residue indices in the CSV file map 1:1 to mmCIF
                # seq_ids. Verify by checking the residue names in the ihm_cross_link_list
                # in the output mmCIF. You may need to add an offset or otherwise map
                # the residue indices, because it looks off to me.
                residue_pair = ihm.restraint.ExperimentalCrossLink(
                    residue1=entity1.residue(i+1), residue2=entity2.residue(j+1))
                # This takes a list of all ambiguous cross-links. Here we're saying there
                # is no ambiguity.
                crosslink_restraint.experimental_cross_links.append([residue_pair])
                residue_pair_restraint = ihm.restraint.ResidueCrossLink(experimental_cross_link=residue_pair,
                                                                        asym1=asym1,
                                                                        asym2=asym2,
                                                                        psi=(1 - fdr),
                                                                        distance=distance)
                crosslink_restraint.cross_links.append(residue_pair_restraint)

    system.restraints.append(crosslink_restraint)

    all_datasets = ihm.dataset.DatasetGroup((crosslink_dataset,))


    system.complete_assembly.name = args.protein_name
    system.complete_assembly.description = 'Integrative model of %s by crosslinking MS and deep learning' % args.protein_name

    protocol = ihm.protocol.Protocol(name='AlphaLink2')
    protocol.steps.append(ihm.protocol.Step(
        assembly=system.complete_assembly, dataset_group=all_datasets, software=alphalink_software,
        method='AlphaLink2', name='AlphaLink2',
        num_models_begin=0, num_models_end=1, multi_scale=False, ensemble=False))

    rep = ihm.representation.Representation(
            [ihm.representation.AtomicSegment(asym, rigid=False)
            for asym in system.asym_units])

    for state_group in system.state_groups:
            for state in state_group:
                for model_group in state:
                    for model in model_group:
                        if not model.assembly:
                            model.assembly = system.complete_assembly
                        model.protocol = protocol
                        if not model.representation:
                            model.representation = rep

    with open(args.output, "w", encoding="utf-8") as fh:
        ihm.dumper.write(fh, [system])

if __name__ == "__main__":
    main()
