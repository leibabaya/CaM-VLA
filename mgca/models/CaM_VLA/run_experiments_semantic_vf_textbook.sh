#!/bin/bash

# 1. 随机种子 (外层循环)
SEEDS=(8192 4096 2048 1024 42)

# 2. 语义温度 (用于 Loss_CLS 的 Softmax T)
SEMANTIC_TEMPS=(10.0)

# 3. VF 对比温度 (用于 Loss_VF 内部的对比约束)
TAU_VFS=(0.9 0.7 0.5 0.3 0.1)

# 4. VF 损失总权重 (用于 Loss_Total 的 lambda)
LAMBDA_VFS=(1.0)


PYTHON_SCRIPT="mgca_module_semantic_vf_textbook.py"
BASE_COMMAND="python $PYTHON_SCRIPT --gpus 2 --strategy ddp"
# --------------------

echo "=========================================================="
echo "   启动 【VF对齐 + 语义分类】 四变量网格搜索实验     "
echo "=========================================================="

for SEED in "${SEEDS[@]}"
do
    echo ""
    echo "############################################################"
    echo "##               正在处理 SEED = ${SEED}                  ##"
    echo "############################################################"

    for SEM_TEMP in "${SEMANTIC_TEMPS[@]}"
    do

        for TAU_VF in "${TAU_VFS[@]}"
        do

            for L_VF in "${LAMBDA_VFS[@]}"
            do
                EXPERIMENT_NAME="s${SEED}_st${SEM_TEMP}_tau${TAU_VF}_lvf${L_VF}"

                echo ""
                echo "--------------------------------------------------------"
                echo ">> 正在执行: Seed=${SEED}, TauVF=${TAU_VF}, LambdaVF=${L_VF}"
                echo ">> Experiment Name: ${EXPERIMENT_NAME}"
                echo "--------------------------------------------------------"

                CUDA_VISIBLE_DEVICES=2,3 ${BASE_COMMAND} \
                    --lambda_cls=1.0 \
                    --seed=${SEED} \
                    --semantic_temperature=${SEM_TEMP} \
                    --tau_vf_global=${TAU_VF} \
                    --lambda_vf=${L_VF} \
                    --experiment_name=${EXPERIMENT_NAME}

                echo ">> 实验 ${EXPERIMENT_NAME} 已完成。"

            done # 结束 LAMBDA_VFS 循环
        done # 结束 TAU_VFS 循环
    done # 结束 SEMANTIC_TEMPS 循环

    echo ""
    echo "## Seed ${SEED} 的所有实验已完成。 ##"

done # 结束 SEEDS 循环

echo ""
echo "=========================================================="
echo "             所有四变量网格搜索实验均已完成！             "
echo "=========================================================="