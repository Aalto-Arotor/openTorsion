class PI():
  """
  This class creates a discrete PI controller for a system.

  Attributes
  ----------
  Kp : float
      Proportional gain
  Ki : float
      Integral gain
  dt : float
      Time step
  setpoint : float
      Desired system output
  limit : float
      Output limit for integrator and controller
  """

  def __init__(self, Kp, Ki, dt, setpoint, limit):
    """
    Parameters
    ----------
    Kp : float
        Proportional gain
    Ki : float
        Integral gain
    dt : float
        Time step
    setpoint : float
        Desired output
    limit : float
        Output limit for integrator and controller
    """
    self.Kp = Kp
    self.Ki = Ki
    self.dt = dt
    self.setpoint = setpoint
    self.limit = limit
    self.integral_error = 0


  def next_step(self, x):
    """
    Calculates the next controller output, taking into account the controller gains and limit.

    Parameters
    ----------
    x : float
        Current system output

    Returns
    -------
    float
        Controller output
    """
    error = self.setpoint - x
    self.integral_error += error*self.Ki*self.dt
    out = self.integral_error + error*self.Kp

    if self.integral_error > self.limit:
        self.integral_error = self.limit

    if out > self.limit:
        return self.limit

    return out