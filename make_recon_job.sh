#!/usr/bin/env bash
# Write a cluster job that collects shape-reconstruction shards with one checkpoint of one run.
#   bash make_recon_job.sh NAME RUN_DIR EPOCH ["EXTRA task flags"]
#   e.g. bash make_recon_job.sh baseline baseline-binary-split4 4500
#        bash make_recon_job.sh parkfixed park-ds20-fixed-split4 4000 "+task.env.touch_model=park +task.env.park_E=338e3 +task.env.park_log_eps=0.003 +task.env.park_calib=False"
# The job finds runs/RUN_DIR/*/nn/*_ep_EPOCH_rew_*.pth on the drive, runs collect_recon.sh with 2000 envs per
# round, and writes shards to /data/recon/NAME_epEPOCH/ and its log to /data/logs/recon_NAME.log.
# EPOCH may be "wait": the job is submitted now (so it can queue) and, once running, waits up to 2 h for the
# file /data/recon/NAME.epoch, e.g. written with
#   kubectl -n erl-ucsd-supp exec job/anderson-softsyn-recon-NAME -- bash -c 'echo 4000 > /data/recon/NAME.epoch'
# If no epoch arrives in 2 h the job exits and frees the GPU.
# MEM (default 24Gi) and SHM (default 16Gi) set the memory request and /dev/shm size, e.g. MEM=16Gi SHM=8Gi.
set -eo pipefail
NAME=$1; RUN=$2; EPOCH=$3; EXTRA=${4:-}
[ -n "$NAME" ] && [ -n "$RUN" ] && [ -n "$EPOCH" ] || { sed -n 2,7p "$0"; exit 1; }
out=k8s/job-softsyn-recon-$NAME.yaml
cat > $out <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: anderson-softsyn-recon-$NAME
  namespace: erl-ucsd-supp
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 43200        # 12 h
  template:
    spec:
      restartPolicy: Never
      tolerations:
      - key: "nautilus.io/reservation"
        operator: "Equal"
        value: "erl-ucsd"
        effect: "NoSchedule"
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values: ["NVIDIA-GeForce-RTX-3090", "NVIDIA-GeForce-RTX-4090", "NVIDIA-A10", "NVIDIA-A40", "NVIDIA-L40", "NVIDIA-L40S", "NVIDIA-RTX-A5000", "NVIDIA-RTX-A6000", "NVIDIA-L4", "NVIDIA-A100-PCIE-40GB", "NVIDIA-A100-80GB-PCIe", "NVIDIA-A100-SXM4-80GB", "Tesla-V100-SXM2-32GB", "NVIDIA-TITAN-RTX", "Quadro-RTX-6000", "Quadro-RTX-8000"]
              - key: kubernetes.io/hostname
                operator: NotIn
                values: ["nrp-01.laccd.edu", "ry-gpu-02.sdsc.optiputer.net", "ry-gpu-08.sdsc.optiputer.net"]
      containers:
      - name: main
        image: nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
        command: ["bash", "-c"]
        args:
        - |
          set -eo pipefail
          export PYTHONUNBUFFERED=1
          source /data/softsyn_env.sh
          cd /data/soft-synesthesia && git pull --ff-only && git log --oneline -1
          mkdir -p /data/logs /data/recon
          E=$EPOCH
          if [ "\$E" = wait ]; then
            f=/data/recon/$NAME.epoch; n=0
            until [ -s \$f ]; do
              n=\$((n + 1)); [ \$n -gt 120 ] && { echo "no \$f after 2 h, exiting to free the GPU"; exit 1; }
              echo "waiting for \$f (minute \$n)"; sleep 60
            done
            E=\$(tr -d '[:space:]' < \$f)
          fi
          echo "epoch: \$E"
          CKPT=\$(ls /data/soft-synesthesia/runs/$RUN/*/nn/*_ep_\${E}_rew_*.pth | head -1)
          echo "checkpoint: \$CKPT"
          test -s "\$CKPT"
          cp "\$CKPT" /data/recon/ckpt_${NAME}_ep\$E.pth
          OUT=/data/recon/${NAME}_ep\$E ENVS=2000 EXTRA="$EXTRA" \\
            bash collect_recon.sh /data/recon/ckpt_${NAME}_ep\$E.pth 2>&1 | tee /data/logs/recon_$NAME.log
          echo "=== DONE"
        resources:
          limits:
            nvidia.com/gpu: 1
            cpu: "4"
            memory: ${MEM:-24Gi}
          requests:
            nvidia.com/gpu: 1
            cpu: "4"
            memory: ${MEM:-24Gi}
        volumeMounts:
        - name: dshm
          mountPath: /dev/shm
        - name: data
          mountPath: /data
      volumes:
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: ${SHM:-16Gi}
      - name: data
        persistentVolumeClaim:
          claimName: anderson-softsyn-data
YAML
echo "wrote $out"
grep -nE "name: anderson|ls /data|OUT=|EXTRA=" $out
