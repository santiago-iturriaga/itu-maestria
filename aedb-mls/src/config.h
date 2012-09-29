#ifndef CONFIG_H_
#define CONFIG_H_

#define MLS__MAX_THREADS        64

#define AGA__PROCESS_RANK       0
#define AGA__NEW_SOL_MSG        0
#define AGA__EXIT_MSG           1
#define AGA__MAX_ARCHIVE_SIZE   2
//#define AGA__MAX_ARCHIVE_SIZE   50

extern int world_rank, world_size;
extern char machine_name[180];

#endif
