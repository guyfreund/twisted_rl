import os
import argparse
import paramiko
import traceback
import stat
from scp import SCPClient
from typing import Optional


class ServerDataManager:
    def __init__(self, override: bool, exclude_extensions: Optional[list[str]] = None):
        self.servers = {
            'server': os.environ.get('SERVER'),
            'server_new': os.environ.get('SERVER_NEW'),
        }
        self.username = os.environ.get('USERNAME')
        self.password = os.environ.get('PASSWORD')
        self.ssh_clients = {}
        self.override = override
        self.dirs_not_to_copy = ['replay_buffer_files', 'her_replay_buffer_files', 'wandb', 'exceptions', 'env']
        self.exclude_extensions = exclude_extensions if exclude_extensions else []

    def connect(self, server: str):
        """Establish an SSH connection to the specified server."""
        if server not in self.servers or not self.servers[server]:
            raise ValueError(f"Server '{server}' not found in configuration.")

        hostname = self.servers[server]

        ssh = paramiko.SSHClient()
        ssh.load_system_host_keys()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username=self.username, password=self.password)

        self.ssh_clients[server] = ssh

    def download(self, sftp: paramiko.SFTPClient, scp: SCPClient, remote_path: str, local_path: str):
        """
        Recursively download a remote directory/file to a local directory/file.

        Args:
            sftp (SFTPClient): The SFTP client to use for listing remote directories.
            scp (SCPClient): The SCP client to use for downloading files.
            remote_path (str): The remote path to download.
            local_path (str): The local path to download to.
        """
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        try:
            if not stat.S_ISDIR(sftp.stat(remote_path).st_mode):
                print(f"Copying file from {remote_path} to {local_path}")
                basename = os.path.basename(remote_path)
                if os.path.exists(local_path) and self.override:
                    os.remove(local_path)
                local_path = local_path.replace(basename, '')
                scp.get(remote_path, local_path)
                return
        except FileNotFoundError as e:
            print(f"Remote path {remote_path} not found.")
            traceback.print_exception(type(e), e, e.__traceback__)
            raise e

        os.makedirs(local_path, exist_ok=True)

        for item in sftp.listdir_attr(remote_path):
            remote_item_path = f"{remote_path}/{item.filename}"
            local_item_path = os.path.join(local_path, item.filename)

            if item.filename in self.dirs_not_to_copy:
                continue

            if stat.S_ISDIR(item.st_mode):  # Corrected usage
                self.download(sftp, scp, remote_item_path, local_item_path)
            else:
                if len(self.exclude_extensions) > 0:
                    if any(item.filename.endswith(ext) for ext in self.exclude_extensions):
                        print(f"Skipping file {item.filename} due to excluded extensions.")
                        continue
                if os.path.exists(local_item_path) and self.override:
                    os.remove(local_item_path)
                print(f"Copying file from {remote_item_path} to {local_item_path}")
                scp.get(remote_item_path, local_item_path)

    def copy_data_from_server(self, server: str, remote_path: str, local_path: str):
        """Copy data from the remote server to the local machine."""
        if server not in self.ssh_clients:
            self.connect(server)

        ssh = self.ssh_clients[server]
        sftp = ssh.open_sftp()  # Open SFTP connection
        with SCPClient(ssh.get_transport()) as scp:
            try:
                self.download(sftp=sftp, scp=scp, remote_path=remote_path, local_path=local_path)
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                raise e
            finally:
                sftp.close()
                scp.close()
                self.close_connection(server)

    def close_connection(self, server: str):
        """Close the SSH connection to the specified server."""
        if server in self.ssh_clients:
            self.ssh_clients[server].close()
            del self.ssh_clients[server]

    def close_all_connections(self):
        """Close all SSH connections."""
        for server in list(self.ssh_clients.keys()):
            self.close_connection(server)


SERVER_TO_PREFIX = {
    'mac': '/Users/guyfreund/projects',
    'server': '/home/g/guyfreund',
    'server_new': '/home/guyfreund',
}


def get_local_path_from_local_and_remote_servers(local_server: str, remote_server: str, remote_path: str, local_path: Optional[str] = None):
    assert local_server != remote_server, "Local and remote servers should be different."
    assert local_server in ['mac', 'server', 'server_new'], f"Local server '{local_server}' not found in configuration."
    assert remote_server in ['mac', 'server', 'server_new'], f"Remote server '{remote_server}' not found in configuration."

    local_prefix = SERVER_TO_PREFIX[local_server]
    remote_prefix = SERVER_TO_PREFIX[remote_server]

    if local_path is not None:
        assert local_path.startswith(local_prefix), f"Local path '{local_path}' does not start with '{local_prefix}'"
    else:
        local_path = remote_path.replace(remote_prefix, local_prefix)

    return local_path


def main(override: bool, local_server: str, remote_server: str, remote_paths: list[str], local_paths: Optional[list[str]] = None, exclude_extensions: Optional[list[str]] = None):
    if local_paths is None:
        local_paths = [None] * len(remote_paths)
    manager = ServerDataManager(override=override, exclude_extensions=exclude_extensions)
    try:
        for local_path, remote_path in zip(local_paths, remote_paths):
            local_path = get_local_path_from_local_and_remote_servers(local_server=local_server, remote_server=remote_server, local_path=local_path, remote_path=remote_path)
            if os.path.exists(local_path) and not override:
                continue
            manager.copy_data_from_server(server=remote_server, remote_path=remote_path, local_path=local_path)
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        raise e
    finally:
        manager.close_all_connections()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-ls', '--local_server', type=str, default='mac', choices=['mac', 'server', 'server_new'])
    arg_parser.add_argument('-rs', '--remote_server', type=str, default='server', choices=['mac', 'server', 'server_new'])
    arg_parser.add_argument('-r', '--remote_paths', type=str, nargs='+', default=None)
    arg_parser.add_argument('-l', '--local_paths', type=str, nargs='+', default=None)
    arg_parser.add_argument('-e', '--exclude_extensions', type=str, nargs='+', default=None)
    arg_parser.add_argument('--not_override', dest='override', action='store_false')
    args = arg_parser.parse_args()
    main(override=args.override, local_server=args.local_server, remote_server=args.remote_server,
         remote_paths=args.remote_paths, local_paths=args.local_paths, exclude_extensions=args.exclude_extensions)
