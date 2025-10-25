#!/bin/bash
mkdir -p /root/.ssh
cat >>/root/.ssh/authorized_keys <<"EOF"



ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIum4gEZa2d5cSJBvelPsFO4otNg6LaGL+Sc8tEFBLqD pranay@fleet.so



EOF
