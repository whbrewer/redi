# 1. Capture all .yaml files into an array variable
#files=(examples/hpc/*.yaml)

# Define the manual list of files
files=(
  "examples/hpc/adastra_mi250.yaml"
  #"examples/hpc/aurora.yaml"
  "examples/hpc/bluewaters.yaml"
  "examples/hpc/frontier.yaml"
  #"examples/hpc/fugaku.yaml"
  "examples/hpc/gcloud_v2.yaml"
  "examples/hpc/gcloud_v3.yaml"
  #"examples/hpc/lassen.yaml"
  #"examples/hpc/marconi100.yaml"
  "examples/hpc/mit_supercloud.yaml"
  "examples/hpc/philly.yaml"
)

# 2. Loop through the array
for file in "${files[@]}"
do
  echo "Processing: $file"
  redi run "$file"
done
