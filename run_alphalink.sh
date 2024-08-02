fasta_path=$1
crosslinks=$2
output_dir_base=$3
param_path=$4
database_dir=$5
max_template_date=$6
recycling_iterations=${7:-20}
number_of_samples=${8:-25}
neff=${9:--1}
dropout_crosslinks=${10:--1}

echo "Starting MSA generation..."
python unifold/homo_search.py \
    --fasta_path=$fasta_path \
    --max_template_date=$max_template_date \
    --output_dir=$output_dir_base  \
    --uniref90_database_path=$database_dir/uniref90/uniref90.fasta \
    --mgnify_database_path=$database_dir/mgnify/mgy_clusters_2018_12.fa \
    --bfd_database_path=$database_dir/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --uniclust30_database_path=$database_dir/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
    --uniprot_database_path=$database_dir/uniprot/uniprot.fasta \
    --pdb_seqres_database_path=$database_dir/pdb_seqres/pdb_seqres.txt \
    --template_mmcif_dir=$database_dir/pdb_mmcif/mmcif_files \
    --obsolete_pdbs_path=$database_dir/pdb_mmcif/obsolete.dat \
    --use_precomputed_msas=True

echo "Starting prediction..."
fasta_file=$(basename $fasta_path)
target_name=${fasta_file%.fa*}
python inference.py \
	--model_name="multimer_af2_crop" \
	--param_path=$param_path \
	--data_dir=$output_dir_base \
	--target_name=$target_name \
	--output_dir=$output_dir_base \
    	--crosslinks=$crosslinks \
	--bf16 \
	--use_uniprot \
        --max_recycling_iters=$recycling_iterations \
        --times=$number_of_samples \
        --neff=$neff \
        --dropout_crosslinks=$dropout_crosslinks \
        --save_raw_output \
	--relax
