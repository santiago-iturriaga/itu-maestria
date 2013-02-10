CC=gcc -Wall -std=c99
CX=g++ -Wall 
LIBS=-lpthread -lrt 
#-mtune=opteron -mfpmath=sse -m64

H_FILES=
CPP_FILES=src/main.cpp src/load_params.cpp src/load_instance.cpp src/etc_matrix.cpp src/energy_matrix.cpp \
	src/solution.cpp src/utils.cpp src/basic/mct.cpp src/basic/minmin.cpp src/basic/pminmin.cpp src/random/cpu_rand.cpp \
	src/random/cpu_drand48.cpp src/random/cpu_mt.cpp src/pals/archivers/aga.cpp src/pals/me_mls_cpu.cpp src/pals/me_rpals_cpu.cpp 

all: d_pals-aga-mct-mt pals-aga-mct-mt pals-aga-mct-randr pals-aga-mct-drand48r d_pals-aga-minmin-mt pals-aga-minmin-mt \
	d_pals-aga-pminmin-mt pals-aga-pminmin-mt pals-adhoc-mct-mt pals-adhoc-mct-randr pals-adhoc-mct-drand48r pals-adhoc-minmin-mt \
	pals-adhoc-pminmin-mt verificador fp

debug: d_pals-aga-mct-mt verificador fp

fp: fp_2obj.cpp
	$(CXX) fp_2obj.cpp -o bin/fp_2obj

verificador: verificador.c
	$(CC) verificador.c -o bin/verificador

d_pals-aga-mct-mt: $(CPP_FILES) $(H_FILES)
	$(CX) -O3 -DOUTPUT_SOLUTION=0 -DARCHIVER_AGA -DINIT_MCT -DCPU_MERSENNE_TWISTER \
		$(CPP_FILES) $(LIBS) -o bin/d_me-mls-aga-mct-mt

pals-aga-mct-mt: $(CPP_FILES) $(H_FILES)
	$(CX) -O3 -DOUTPUT_SOLUTION=1 -DARCHIVER_AGA -DINIT_MCT -DCPU_MERSENNE_TWISTER \
		$(CPP_FILES) $(LIBS) -o bin/me-mls-aga-mct-mt

pals-aga-mct-randr: $(CPP_FILES) $(H_FILES)
	$(CX) -O3 -DOUTPUT_SOLUTION=1 -DARCHIVER_AGA -DINIT_MCT -DCPU_RAND \
		$(CPP_FILES) $(LIBS) -o bin/me-mls-aga-mct-randr

pals-aga-mct-drand48r: $(CPP_FILES) $(H_FILES)
	$(CX) -O3 -DOUTPUT_SOLUTION=1 -DARCHIVER_AGA -DINIT_MCT -DCPU_DRAND48 \
		$(CPP_FILES) $(LIBS) -o bin/me-mls-aga-mct-drand48r

d_pals-aga-minmin-mt: $(CPP_FILES) $(H_FILES)
	$(CX) -O3 -DOUTPUT_SOLUTION=0 -DARCHIVER_AGA -DINIT_MINMIN -DCPU_MERSENNE_TWISTER \
		$(CPP_FILES) $(LIBS) -o bin/d_me-mls-aga-minmin-mt

pals-aga-minmin-mt: $(CPP_FILES) $(H_FILES)
	$(CX) -O3 -DOUTPUT_SOLUTION=1 -DARCHIVER_AGA -DINIT_MINMIN -DCPU_MERSENNE_TWISTER \
		$(CPP_FILES) $(LIBS) -o bin/me-mls-aga-minmin-mt

d_pals-aga-pminmin-mt: $(CPP_FILES) $(H_FILES)
	$(CX) -O3 -DOUTPUT_SOLUTION=0 -DARCHIVER_AGA -DINIT_PMINMIN -DCPU_MERSENNE_TWISTER \
		$(CPP_FILES) $(LIBS) -o bin/d_me-mls-aga-pminmin-mt

pals-aga-pminmin-mt: $(CPP_FILES) $(H_FILES)
	$(CX) -O3 -DOUTPUT_SOLUTION=1 -DARCHIVER_AGA -DINIT_PMINMIN -DCPU_MERSENNE_TWISTER \
		$(CPP_FILES) $(LIBS) -o bin/me-mls-aga-pminmin-mt

pals-adhoc-mct-mt: $(CPP_FILES) $(H_FILES)
	$(CX) -O3 -DOUTPUT_SOLUTION=1 -DINIT_MCT -DCPU_MERSENNE_TWISTER \
		$(CPP_FILES) $(LIBS) -o bin/me-mls-adhoc-mct-mt

pals-adhoc-mct-randr: $(CPP_FILES) $(H_FILES)
	$(CX) -O3 -DOUTPUT_SOLUTION=1 -DINIT_MCT -DCPU_RAND \
		$(CPP_FILES) $(LIBS) -o bin/me-mls-adhoc-mct-randr

pals-adhoc-mct-drand48r: $(CPP_FILES) $(H_FILES)
	$(CX) -O3 -DOUTPUT_SOLUTION=1 -DINIT_MCT -DCPU_DRAND48 \
		$(CPP_FILES) $(LIBS) -o bin/me-mls-adhoc-mct-drand48r
		
pals-adhoc-minmin-mt: $(CPP_FILES) $(H_FILES)
	$(CX) -O3 -DOUTPUT_SOLUTION=1 -DINIT_MINMIN -DCPU_MERSENNE_TWISTER \
		$(CPP_FILES) $(LIBS) -o bin/me-mls-adhoc-minmin-mt

pals-adhoc-pminmin-mt: $(CPP_FILES) $(H_FILES)
	$(CX) -O3 -DOUTPUT_SOLUTION=1 -DINIT_PMINMIN -DCPU_MERSENNE_TWISTER \
		$(CPP_FILES) $(LIBS) -o bin/me-mls-adhoc-pminmin-mt

pals-prof: src/main.cpp \
		src/load_params.cpp \
		src/load_instance.cpp \
		src/etc_matrix.cpp \
		src/energy_matrix.cpp \
		src/solution.cpp \
		src/utils.cpp \
		src/basic/mct.cpp \
		src/basic/minmin.cpp \
		src/basic/pminmin.cpp \
		src/random/cpu_rand.cpp \
		src/random/cpu_drand48.cpp \
		src/random/cpu_mt.cpp \
		src/pals/archivers/aga.cpp \
		src/pals/me_mls_cpu.cpp \
		src/pals/me_rpals_cpu.cpp
	$(CX) -g -O3 src/main.cpp \
		src/load_instance.cpp \
		src/load_params.cpp \
		src/etc_matrix.cpp \
		src/energy_matrix.cpp \
		src/solution.cpp \
		src/utils.cpp \
		src/basic/mct.cpp \
		src/basic/minmin.cpp \
		src/basic/pminmin.cpp \
		src/random/cpu_rand.cpp \
		src/random/cpu_drand48.cpp \
		src/random/cpu_mt.cpp \
		src/pals/archivers/aga.cpp \
		src/pals/me_mls_cpu.cpp \
		src/pals/me_rpals_cpu.cpp \
		$(LIBS) -o bin/pals_cpu_prof

pals-gdb: src/main.cpp \
		src/load_params.cpp \
		src/load_instance.cpp \
		src/etc_matrix.cpp \
		src/energy_matrix.cpp \
		src/solution.cpp \
		src/utils.cpp \
		src/basic/mct.cpp \
		src/basic/minmin.cpp \
		src/basic/pminmin.cpp \
		src/random/cpu_rand.cpp \
		src/random/cpu_drand48.cpp \
		src/random/cpu_mt.cpp \
		src/pals/archivers/aga.cpp \
		src/pals/me_mls_cpu.cpp \
		src/pals/me_rpals_cpu.cpp 
	$(CX) -g src/main.cpp \
		src/load_instance.cpp \
		src/load_params.cpp \
		src/etc_matrix.cpp \
		src/energy_matrix.cpp \
		src/solution.cpp \
		src/utils.cpp \
		src/basic/mct.cpp \
		src/basic/minmin.cpp \
		src/basic/pminmin.cpp \
		src/random/cpu_rand.cpp \
		src/random/cpu_drand48.cpp \
		src/random/cpu_mt.cpp \
		src/pals/archivers/aga.cpp \
		src/pals/me_mls_cpu.cpp \
		src/pals/me_rpals_cpu.cpp \
		$(LIBS) -o bin/$(OUTPUT_BIN) 

clean:
	rm bin/$(OUTPUT_BIN)
