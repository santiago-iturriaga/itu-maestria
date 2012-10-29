#ifndef CONFIG_H_
#define CONFIG_H_

#ifdef LOCAL
    //#define NS3_BIN "/home/santiago/itu-maestria/svn/trunk/aedb-mls/bin/ns3_fake"
    #define NS3_BIN "/home/santiago/itu-maestria/trunk/aedb-mls/bin/ns3_fake"
#else
    #define NS3_BIN "/home/clusterusers/siturriaga/itu-maestria/trunk/aedb-mls/bin/ns3"
#endif

//#define MPI_MODE_STANDARD
//#define MPI_MODE_SYNC
#define MPI_MODE_BUFFERED
//#define MLS__BUFFER_SIZE        10
#define MLS__BUFFER_SIZE        100

//#define MLS__MAX_THREADS        1
#define MLS__MAX_THREADS        64

#define AGA__PROCESS_RANK       0
#define AGA__NEW_SOL_MSG        0
#define AGA__REQ_SOL_MSG        1
#define AGA__EXIT_MSG           2
//#define AGA__MAX_ARCHIVE_SIZE   5
#define AGA__MAX_ARCHIVE_SIZE   50

// How many local search operators
#define NUM_LS_OPERATORS 4

// Local search operators
#define LS_ENERGY 0
#define LS_COVERAGE 1
#define LS_FORWARDING 2
#define LS_TIME 3

extern int world_rank, world_size;
extern char machine_name[180];

#endif
