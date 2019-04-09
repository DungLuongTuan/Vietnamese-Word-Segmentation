class LM(object):
    def __init__(self, embed):
        self.embed = embed

    def get(self, w):
        if (w in self.embed.keys()):
            return self.embed[w]
        return self.embed["<OOV>"]