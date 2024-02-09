import requests
from io import BytesIO
import pandas

# https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_language_modeling.ipynb#scrollTo=f2c05017

query_url = "https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Csequence%2Ccc_subcellular_location&format=tsv&query=%28%28organism_id%3A9606%29%20AND%20%28reviewed%3Atrue%29%20AND%20%28length%3A%5B80%20TO%20500%5D%29%29"
uniprot_request = requests.get(query_url)
bio = BytesIO(uniprot_request.content)
df = pandas.read_csv(bio, compression="gzip", sep="\t")
df.dropna(inplace=True)  # Drop proteins with missing columns
# Make one dataframe of proteins that contain `cytosol` or `cytoplasm` in their subcellular localization column,
# and a second that mentions the `membrane` or `cell membrane`. To ensure we don't get overlap, we ensure each
# dataframe only contains proteins that don't match the other search term.
cytosolic = df["Subcellular location [CC]"].str.contains("Cytosol") | df[
    "Subcellular location [CC]"
].str.contains("Cytoplasm")
membrane = df["Subcellular location [CC]"].str.contains("Membrane") | df[
    "Subcellular location [CC]"
].str.contains("Cell membrane")

cytosolic_df = df[cytosolic & ~membrane]
membrane_df = df[membrane & ~cytosolic]

df.to_parquet("../data/uniprot.parquet", compression="snappy", index=False)
df.to_parquet("../data/cytosolic.parquet", compression="snappy", index=False)
df.to_parquet("../data/membrane.parquet", compression="snappy", index=False)
