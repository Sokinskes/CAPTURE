FROM ros:noetic-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-pip \
    python3-catkin-tools \
    python3-rosdep \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Conan for ros_kortex build
RUN python3 -m pip install --no-cache-dir conan==1.59 && \
    conan config set general.revisions_enabled=1 && \
    conan profile new default --detect >/dev/null && \
    conan profile update settings.compiler.libcxx=libstdc++11 default

# Init rosdep
RUN rosdep init || true
RUN rosdep update

# Build ros_kortex
RUN mkdir -p /opt/catkin_ws/src
WORKDIR /opt/catkin_ws/src
RUN git clone -b noetic-devel https://github.com/Kinovarobotics/ros_kortex.git
WORKDIR /opt/catkin_ws
RUN rosdep install --from-paths src --ignore-src -y
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Copy CAPTURE repo
WORKDIR /workspace/CAPTURE
COPY . /workspace/CAPTURE

# Python deps
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Default entry
SHELL ["/bin/bash", "-c"]
CMD source /opt/ros/noetic/setup.bash && source /opt/catkin_ws/devel/setup.bash && bash
