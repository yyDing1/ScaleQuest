FROM nvcr.io/nvidia/pytorch:24.04-py3

RUN sed -i "s|archive.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g" /etc/apt/sources.list \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone

RUN apt update \
    && apt install build-essential openssh-server net-tools psmisc tzdata tmux lsof libaio-dev -y \
    && service ssh start

RUN ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -P "" \
    && cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys \
    && echo "StrictHostKeyChecking no" > ~/.ssh/config \
    && chmod 600 ~/.ssh/{config,authorized_keys}

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt \
    && pip install flash-attn --no-build-isolation \
    && pip install -U nvitop

ENTRYPOINT ["/usr/sbin/sshd", "-D"]
