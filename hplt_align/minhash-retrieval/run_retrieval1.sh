PKL=$1
HPLT_DIR=$2
OUTD=$3

TEXTFIELD=$4  # 'text', 't' for outputs from stage2
FILEPATTERN=$5 # '*.zst', 'text.zst' for outputs from stage2

echo "Running retrieval:" "$FILEPATTERN"

mkdir -p $OUTD
find $HPLT_DIR -name "$FILEPATTERN" | parallel --eta --joblog $OUTD/joblog --linebuffer -j 100 "python -u retriever.py retrieve {} -t $TEXTFIELD --fcontent=$PKL >$OUTD/{#}.out 2>$OUTD/{#}.err"
