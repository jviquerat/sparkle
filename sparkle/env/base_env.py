# Generic imports

###############################################
### Base environment
class base_env():

    # Create object
    def __init__(self):
        pass

    # Reset environment
    def reset(self):
        raise NotImplementedError

    # Cost function
    def cost(self, x):
        raise NotImplementedError

    # Rendering
    def render(self, x, c):
        pass

    # Close environment
    def close(self):
        raise NotImplementedError
