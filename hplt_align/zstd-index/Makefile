CC ?= gcc
all:
	${CC} -O3 -DNDEBUG ${CFLAGS} -lzstd ${LDFLAGS} line_offsets_of_seekable_zstd.c zstd-seek.c -o line_offsets_of_seekable_zstd
