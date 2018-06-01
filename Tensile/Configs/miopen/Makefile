P=problems
DEEPBENCH_CONV_1x1=$P/nn/deepbench_conv_1x1_batchN.yml $P/nn/deepbench_conv_1x1_batch1.yml
RESNET=$P/nn/resnet50_batch64.yml

# commonly-used headers and footers:
HEADER=boiler/header.yml
FOOTER=boiler/library_logic_vega10_only.yml

# Override TYPE as sgemm, hgemm (hgemm_hpa, dgemm, etc in future)
TYPE=sgemm

# Override these to provide custom solution level or files
# Explore level 0..9 - higer level indicates more intensive search.
#  Only 3,5,7 supported.
#  Exact solutions are defined in the specified solution file.
EXPLORE_LEVEL=5

# SOLUTION_LARGE is used for large matrix type
SOLUTION_LARGE=solutions/$(TYPE)_large_explore_$(EXPLORE_LEVEL).yml

# SOLUTION_SKINNY is used for problems with skinny M or N
SOLUTION_SKINNY=solutions/$(TYPE)_skinny_explore_$(EXPLORE_LEVEL).yml

# Targets specify different collections of solutions + problems

all: \
	$(TYPE)_deepbench_conv1x1.yaml \
	$(TYPE)_deepbench_gemm_nn.yaml \
	$(TYPE)_deepbench_gemm_nt.yaml \
	$(TYPE)_deepbench_gemm_tn.yaml

# DeepBench Convolution:
$(TYPE)_deepbench_conv1x1.yaml: $(HEADER) types/$(TYPE)_nn.yml \
	$(SOLUTION_SKINNY) $(DEEPBENCH_CONV_1x1) \
	$(FOOTER)

# DeepBench GEMM
$(TYPE)_deepbench_gemm_nn.yaml: $(HEADER) types/$(TYPE)_nn.yml \
	$(SOLUTION_SKINNY) $P/nn/deepbench_gemm_skinny.yml \
	$(SOLUTION_LARGE) $P/nn/deepbench_gemm_large.yml \
	$(FOOTER)
$(TYPE)_deepbench_gemm_nt.yaml: $(HEADER) types/$(TYPE)_nt.yml \
	$(SOLUTION_LARGE) $P/nt/deepbench_gemm_large.yml \
	$(SOLUTION_SKINNY) $P/nt/deepbench_gemm_skinny.yml \
	$(FOOTER)
$(TYPE)_deepbench_gemm_tn.yaml: $(HEADER) types/$(TYPE)_tn.yml \
	$(SOLUTION_LARGE) $P/tn/deepbench_gemm_large.yml \
	$(SOLUTION_SKINNY) $P/tn/deepbench_gemm_skinny.yml \
	$(FOOTER)


.PHONY: clean
clean:
	rm -f *.yaml

# Common rule used for all 
%.yaml:
	cat $^ > $@
