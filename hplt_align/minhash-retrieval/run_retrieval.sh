PKL=$1
HPLT_DIR=$2
OUTD=$3
mkdir -p $OUTD
find $HPLT_DIR -name "*.zst"|parallel --eta --joblog $OUTD/joblog1 --linebuffer -j 50 ./index_lines.sh {}|parallel --joblog $OUTD/joblog2 -j 50 --pipe --round-robin "python -u retriever.py retrieve --fcontent=$PKL >$OUTD/{#}.out 2>$OUTD/{#}.err"
