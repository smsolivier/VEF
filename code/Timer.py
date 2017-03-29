import time

''' Class for timing functions ''' 

class timer:

	''' Object that tracks start and end time and prints elapsed time 
		Instantiating timer starts the clock and timer.stop() stops the clock 
	'''

	def __init__(self, out='Wall Time ='):
		self.start = time.time()
		self.out = out

	def stop(self):
		elapsed = time.time() - self.start

		fmt = '{:.2f}'
		str = ''
		if (elapsed < 60):
			str = fmt.format(elapsed) + ' seconds'

		elif (elapsed < 3600):
			str = fmt.format(elapsed/60) + ' minutes'

		elif (elapsed > 3600):
			str = fmt.format(elapsed/3600) + ' hours'

		print(self.out, str)

		return elapsed, str 
