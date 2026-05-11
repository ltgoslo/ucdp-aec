#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zstd-seek.h"


#ifndef NREAD
#define NREAD 100
#endif


bool isLittleEndian(){
    volatile int x = 1;
    return *(char*)(&x) == 1;
}

void writeLE32(char **output, uint32_t data){
    if(isLittleEndian()){
        memcpy(*output, &data, sizeof(data));
    } else {
        uint32_t swap = ((data & 0xFF000000) >> 24) |
                        ((data & 0x00FF0000) >> 8)  |
                        ((data & 0x0000FF00) << 8)  |
                        ((data & 0x000000FF) << 24);
        memcpy(*output, &swap, sizeof(swap));
    }
	*output += sizeof(uint32_t);
}

int main(int argc, char **argv){
	if(argc != 2){
		fprintf(stderr, "format: %s path/to/seekable.file.zstd\n", argv[0]);
		return EXIT_FAILURE;
	}

	ZSTDSeek_Context *sctx = ZSTDSeek_createFromFile(argv[1]);
	if(!sctx){
		fprintf(stderr, "Can't open zstd file.\n");
		return EXIT_FAILURE;
	}

	ZSTDSeek_JumpTable *jt = ZSTDSeek_getJumpTableOfContext(sctx);
	if(!jt){
		fprintf(stderr, "Zstd file not seekable.\n");
		return EXIT_FAILURE;
	}

	size_t max_buffer = 0;
	uint64_t last_uncompressed = 0;
	for(ZSTDSeek_JumpTableRecord *record = jt->records; record != jt->records + jt->length; ++record){
		size_t candidate = record->uncompressedPos - last_uncompressed;
		if(max_buffer < candidate)
			max_buffer = candidate;
		last_uncompressed = record->uncompressedPos;
	}

	size_t output_size = strlen(argv[1]) + 1 + sizeof(uint32_t)*(1+2*(jt->length-1));
	char *output_start = malloc(output_size);
	char *output_end = output_start + output_size;
	char *output = output_start;

	strcpy(output, argv[1]);
	output += strlen(argv[1]) + 1;
	writeLE32(&output, jt->length-1);

	char *buffer = malloc(max_buffer * NREAD);
	for(ZSTDSeek_JumpTableRecord *brecord = jt->records; brecord < jt->records + jt->length - 1; brecord += NREAD){
		assert(ZSTDSeek_tell(sctx) == brecord->uncompressedPos);
		assert(output < output_end);
		ZSTDSeek_JumpTableRecord *end = brecord+NREAD<jt->records+jt->length-1?brecord+NREAD:jt->records+jt->length-1;

		size_t read_size = 0;
		for(ZSTDSeek_JumpTableRecord *record = brecord; record < end; ++record)
			read_size += record[1].uncompressedPos - record->uncompressedPos;
		size_t read = ZSTDSeek_read(buffer, read_size, sctx);

		size_t buffer_offset = 0;
		for(ZSTDSeek_JumpTableRecord *record = brecord; record < end; ++record){
			uint32_t compressed_size = record[1].compressedPos - record->compressedPos;
			uint32_t uncompressed_size = record[1].uncompressedPos - record->uncompressedPos;

			uint32_t newlines = 0;
			for(size_t i = buffer_offset; i < buffer_offset+uncompressed_size; ++i)
				if(buffer[i] == '\n')
					++newlines;
			writeLE32(&output, newlines);
			writeLE32(&output, compressed_size);

			buffer_offset += uncompressed_size;
		}
	}

	assert(output == output_end);

	char *output_name = malloc(strlen(argv[1]) + sizeof(".zindex"));
	strcpy(output_name, argv[1]);
	strcat(output_name, ".zindex");

	FILE *output_file = fopen(output_name, "wb");
	size_t r = fwrite(output_start, output_size, 1, output_file);
	fclose(output_file);

	free(output_name);
	free(buffer);
	ZSTDSeek_free(sctx);
	return EXIT_SUCCESS;
}
