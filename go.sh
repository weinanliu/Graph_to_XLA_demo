#!/bin/bash
set -x


rm -rf graph_test xla_test

export TF_DUMP_GRAPH_PREFIX=$(pwd)/graph_test
export TF_XLA_FLAGS="--tf_xla_clustering_debug"
export XLA_FLAGS="\
        --xla_dump_to=$(pwd)/xla_test \
        --xla_dump_hlo_as_text \
        --xla_dump_hlo_as_html \
        --xla_dump_hlo_as_proto \
        --xla_eliminate_hlo_implicit_broadcast \
        "
        #--xla_dump_hlo_as_dot \
        #--xla_dump_fusion_visualization \
        #--xla_hlo_graph_sharding_color \
        #--xla_backend_extra_options=123 \
        #--xla_gpu_asm_extra_flags=456


#./train.py

./make_test_graph.py

## $(pwd)文件夹放在tensorflow源码根目录下
#bazel build \
#	--config=dbg \
#	--copt=-Wno-gnu-offsetof-extensions \
#	--verbose_failures \
#	test_graph my_binary

