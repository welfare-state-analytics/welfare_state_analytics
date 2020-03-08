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

network_name = os.environ['DOCKER_NETWORK_NAME']

def read_userlist():
    whitelist, admin = set(), set()
    filename = os.path.join(os.path.dirname(__file__), "userlist")
    if os.path.isfile(filename):
        with open(filename, "r") as fi:
            lines = [
                x.split() for x in [ y.strip() for y in fi.readlines() ]
                    if len(x) > 0 and not x.startswith('#')
            ]
        whitelist = set([ x[0] for x in lines ])
        admin = set([ x[0] for x in lines if len(x) > 1 and x[1] == "admin" ])
    return whitelist, admin

c = get_config()

## Generic
c.JupyterHub.admin_access = True
c.Spawner.default_url = '/lab'

c.JupyterHub.authenticator_class = oauthenticator.github.GitHubOAuthenticator

c.GitHubOAuthenticator.oauth_callback_url = os.environ['OAUTH_CALLBACK_URL']
c.GitHubOAuthenticator.client_id = os.environ['OAUTH_CLIENT_ID']
c.GitHubOAuthenticator.client_secret = os.environ['OAUTH_CLIENT_SECRET']

c.Authenticator.whitelist, c.Authenticator.admin_users = read_userlist()

c.JupyterHub.spawner_class = 'dockerspawner.DockerSpawner'
c.DockerSpawner.image = os.environ['DOCKER_JUPYTER_CONTAINER']
c.DockerSpawner.network_name = network_name
c.JupyterHub.hub_ip = os.environ['HUB_IP']

c.DockerSpawner.use_internal_ip = True
c.DockerSpawner.network_name = network_name
c.DockerSpawner.extra_host_config = { 'network_mode': network_name }

# user data persistence
# see https://github.com/jupyterhub/dockerspawner#data-persistence-and-dockerspawner
notebook_dir = os.environ.get('DOCKER_NOTEBOOK_DIR') or '/home/jovyan'
c.DockerSpawner.notebook_dir = notebook_dir
c.DockerSpawner.volumes = { 'jupyterhub-user-{username}': notebook_dir }

# Other stuff
c.Spawner.cpu_limit = 1
c.Spawner.mem_limit = '10G'

# ssl = join(here, 'ssl')
# keyfile = join(ssl, 'ssl.key')
# certfile = join(ssl, 'ssl.cert')
# if os.path.exists(keyfile):
#     c.JupyterHub.ssl_key = keyfile
# if os.path.exists(certfile):
#     c.JupyterHub.ssl_cert = certfile

## Services
c.JupyterHub.services = [
    {
        'name': 'cull_idle',
        'admin': True,
        'command': 'python /srv/jupyterhub/cull_idle_servers.py --timeout=3600'.split(),
    },
]
