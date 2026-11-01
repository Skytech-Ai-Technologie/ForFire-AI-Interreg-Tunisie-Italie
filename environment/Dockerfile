# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Reproduction image for the Jetson Orin Nano target.
# Reference stack: JetPack 5.1.2 / L4T r35.2.x, CUDA 11.4, TensorRT 8.5,
# Python 3.8.
#
# IMPORTANT
# ---------
# 1. The base image MUST match the JetPack (L4T) release running on the target
#    board, or TensorRT engines built inside the container will not load on the
#    device. Check the board with:  cat /etc/nv_tegra_release
#    and adjust the base tag accordingly.
# 2. PyTorch on Jetson comes from NVIDIA's L4T wheel index, not PyPI. Provide
#    it via --build-arg TORCH_WHEEL_URL=<nvidia wheel url>, or extend a base
#    image that already ships it.
# ---------------------------------------------------------------------------

FROM nvcr.io/nvidia/l4t-tensorrt:r35.2.1-runtime

ARG TORCH_WHEEL_URL=""

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY environment/requirements.lock /tmp/requirements.lock

RUN pip3 install --upgrade pip \
 && if [ -n "$TORCH_WHEEL_URL" ]; then pip3 install "$TORCH_WHEEL_URL"; fi \
 && pip3 install -r /tmp/requirements.lock

COPY . /workspace/uav-fire-detection
RUN pip3 install --no-deps -e /workspace/uav-fire-detection

# Sanity check: print captured environment (versions, JetPack, CUDA).
RUN python3 /workspace/uav-fire-detection/scripts/capture_environment.py --print-only || true

CMD ["bash"]
