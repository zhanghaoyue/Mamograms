
class CustomDict(object):
    def __init__(self, adict):
        """
        Customized class that contain a dictionary to change the print behavior
        to favour skorch.
        """

        self.adict = adict

    def __repr__(self):
        return 'SECRET'





