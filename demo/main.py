class LM(object):
    def __init__(self, embed):
        self.embed = embed

    def get(self, w):
        if (w in self.embed.keys()):
            return self.embed[w]
        return self.embed["<OOV>"]

import pdb
from app.apis.lm import LM
from app import app
app.run(host = '0.0.0.0', port = 5000, debug = True)