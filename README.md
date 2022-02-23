 # MicromagneticAnalysisTools

Code I use for analysing data from MuMax3.


## TODO

- [x] Fix plot colour bug for HSV colours
- [x] Fix animation bug where it doesn't move
- [x] Fix colour bug plog where e.g. m = [-0.005441933153905671, 6.66444601810444e-19, 0.9999851925721442] is magenta instead of white
- [x] Fix bug with quiver plot where step size != 1 messes it up
- [x] Stop it from spamming "Clipping input data..."
- [ ] Allow simultaneous plotting of quiver and colour
- [ ] Delete redundant functions e.g. COM calculation that does not go to a comoving frame
- [ ] Some functions (e.g. `vecToRGB`) should be moved to a separate `utils.py` directory
- [ ] 'Proper' docstrings
- [ ] Skyrmion calculations with `findiff` rather than naive discretisation
- [ ] Sphinx documentation