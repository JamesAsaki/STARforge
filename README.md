# ATX 10x count v1.0

For ATX v4.3 libraries.

By Jim Asaki :)

This repo packages the barcode-counting workflow as the `STARforge` Python distribution with the `atx_tenx_count` command-line entrypoint.

Goal: build a cRNA mapping workflow with fixed offsets, mismatch-tolerant barcode mapping, cell/UMI correction, collision diagnostics, and global collision cleanup.

## Example command

```bash
atx_tenx_count \
  --sample-name sample_01 \
  --r1-fastq /path/to/sample_R1.fastq.gz \
  --r2-fastq /path/to/sample_R2.fastq.gz \
  --cell-barcodes /path/to/barcodes.tsv.gz \
  --mhc-ref /path/to/mhc_ref.csv \
  --pep-ref /path/to/pep_ref.csv \ #or instead use feature id table see below
  --out-dir /path/to/output \
  --skip-graphs
```

## Pipeline

### Stage 1: read parsing

- R1: `cell_bc` (16) + `umi` (12)
- R2: fixed prefix + `mhc_bc` (15) + `TTCC` + `pep_bc` (25) + `ACTC` + `hekumi` (16)
- Strict R2 constants are required by default
- Duplicate observations are aggregated into `reads`
- Filtering is applied on average cell and UMI quality

### Stage 2: feature mapping

- Peptide barcodes are mapped with up to 4 mismatches
- MHC barcodes are mapped with up to 2 mismatches
- Mapping uses a block-based seed index rather than brute-force comparison
- Cell barcodes are mapped with up to 1 mismatch against a user-supplied whitelist
- In practice that whitelist is usually one of:
    - Preferred: map to 10x GEX cellranger barcode list (outfolder -> outs -> raw_feature_bc_matrix -> barcodes.tsv.gz) contains empty droplets (takes an hour)
    - map to 10x GEX cellranger filtered barcode list (outfolder -> outs -> filtered_feature_bc_matrix -> barcodes.tsv.gz) (fastest)
    - map to 10x 5' cell barcode list (takes hours)
    - no mapping, collapse on 1 hamming distance #not implemented
- Exact/mismatch buckets and mapping diagnostics are printed

### Stage 3: within-cell-feature UMI collapse

- Iterative UMI collapse, up to 3 passes by default
- Applied within `(cell_id, mhc_id, pep_id)`

### Stage 4: UMI collision remediation

- Observed `(cell_id, umi)` collision diagnostics and heatmaps
- Global collision resolution on `(cell_id, umi)` using ratio and tie rules

## Reference inputs

You can provide references in one of two ways.

### Option 1: separate peptide and MHC barcode lists

This is for many-on-many libraries (all allele on all peptides).

Peptide list:

```text
pep_id,pep_bc
NYESO_0,NNNNNNNNNNNNNNNNNNNNNNNNN
...
```

MHC list:

```text
mhc_id,mhc_bc
A*02:01,NNNNNNNNNNNNNNN
...
```

### Option 2: feature barcode list

This is the better fit when peptide cloning is allele-specific.

Rules:

- `feature_id` must be `mhc_id|pep_id`
- `feature_bc` must be `[mhc_bc]TTCC[pep_bc]`

Example:

```text
feature_id,feature_bc
A*02:01|NYESO_0,NNNNNNNNNNNNNNNTTCCNNNNNNNNNNNNNNNNNNNNNNNNN
```
