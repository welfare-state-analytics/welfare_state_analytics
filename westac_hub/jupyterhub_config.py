# JupyterHub configuration
#
## If you update this file, do not forget to delete the `jupyterhub_data` volume before restarting the jupyterhub service:
##
##     docker volume rm jupyterhub_jupyterhub_data
##
## or, if you changed the COMPOSE_PROJECT_NAME to <name>:
##
##    docker volume rm <name>_jupyterhub_data
##

import os
import oauthenticator
import netifaces

# try:
#     from jupyter_client.localinterfaces import public_ips
# except:
#     from IPython.utils.localinterfaces  import public_ips

network_name = os.environ['DOCKER_NETWORK_NAME']
spawn_cmd = os.environ.get('DOCKER_SPAWN_CMD', "start-singleuser.sh")
notebook_dir = os.environ.get('DOCKER_NOTEBOOK_DIR') or '/home/humlab'
data_dir = os.environ.get('DATA_VOLUME_CONTAINER', '/data')
services = [
    {
        'name': 'cull_idle',
        'admin': True,
        'command': 'python3 /srv/jupyterhub/cull_idle_servers.py --timeout=3600'.split(),
    },
]

def get_ip():
    try:
        #return public_ips()[0]
        docker0 = netifaces.ifaddresses('docker0')
        docker0_ipv4 = docker0[netifaces.AF_INET][0]
        return docker0_ipv4['addr']
    except:
        return os.environ['HUB_IP']

def read_userlist():
    whitelist, admin = set(), set()
    filename = os.path.join(os.path.dirname(__file__), "userlist")
    assert os.path.isfile(filename), "userlist expected at {}".format(filename)
    with open(filename, "r") as fi:
        lines = [
            x.split() for x in [ y.strip() for y in fi.readlines() ]
                if len(x) > 0 and not x.startswith('#')
        ]
    whitelist = set([ x[0] for x in lines ])
    admin = set([ x[0] for x in lines if len(x) > 1 and x[1] == "admin" ])
    return whitelist, admin

c = get_config()

c.JupyterHub.spawner_class = 'dockerspawner.DockerSpawner'
c.JupyterHub.admin_access = True
#c.JupyterHub.hub_port = 8000                                        # The public facing port of the proxy
c.JupyterHub.hub_ip = '0.0.0.0' #os.environ['HUB_IP'] # get_ip()    # The public facing ip of the whole application (the proxy)
c.JupyterHub.hub_connect_ip = os.environ['HUB_IP']                  # The ip for this process

c.JupyterHub.authenticator_class = oauthenticator.github.GitHubOAuthenticator
c.JupyterHub.cookie_secret_file = os.path.join(data_dir, 'jupyterhub_cookie_secret')
# c.JupyterHub.data_files_path = '/opt/conda/share/jupyterhub'

c.GitHubOAuthenticator.oauth_callback_url = os.environ['OAUTH_CALLBACK_URL']
c.GitHubOAuthenticator.client_id = os.environ['OAUTH_CLIENT_ID']
c.GitHubOAuthenticator.client_secret = os.environ['OAUTH_CLIENT_SECRET']

# c.PAMAuthenticator.open_sessions = False

c.Authenticator.whitelist, c.Authenticator.admin_users = read_userlist()

c.DockerSpawner.image = os.environ['DOCKER_JUPYTER_CONTAINER']
# c.DockerSpawner.extra_create_kwargs.update({ 'command': spawn_cmd })
#c.DockerSpawner.extra_host_config = { 'network_mode': network_name }       # Pass the network name as argument to spawned containers
c.DockerSpawner.use_internal_ip = True
c.DockerSpawner.network_name = network_name
c.DockerSpawner.notebook_dir = '/home/jovyan' # notebook_dir
c.DockerSpawner.volumes = { 'jupyterhub-user-{username}': notebook_dir }
c.DockerSpawner.remove_containers = True                                    # Remove containers once they are stopped
c.DockerSpawner.host_ip = "0.0.0.0"

# c.DockerSpawner.links={network_name: network_name}

c.Spawner.default_url = '/lab'
# c.Spawner.cpu_limit = 1
c.Spawner.mem_limit = '3G'
c.Spawner.notebook_dir = notebook_dir

# Debug settimgs
c.JupyterHub.debug_proxy = True
c.DockerSpawner.debug = True
c.JupyterHub.log_level = 10
c.Spawner.cmd = ['jupyterhub-singleuser', '--debug']
c.Authenticator.auto_login = False
