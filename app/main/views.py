from . import main


@main.route('/test/<username>')
def hello(username):
    return 'service test successful, ' + username
