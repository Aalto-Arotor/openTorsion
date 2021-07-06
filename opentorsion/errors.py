class DOF_mismatch_error(Exception):
    def __init__(self, message):
        self.message = message


if __name__ == "__main__":
    raise DOF_mismatch_error("Mismatched degrees of freedom")
