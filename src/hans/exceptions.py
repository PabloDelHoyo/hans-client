
class HansClientException(Exception):
    pass


class CannotStartRoundException(HansClientException):
    """ Raised when the client cannot start a round. The reason
    for that is probably that the question has not been set"""
