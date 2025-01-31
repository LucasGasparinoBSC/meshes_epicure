# Repo for EPICURE meshes

Repo to provide support for the EPICURE project. Contains:

- A script to generate a TGV mesh using GMSH + gmsh2sod2d
- Some example .geo files

The script invokes `gmsh` to generate the `.msh` file, then runs the `gmsh2sod2d` python script to convert it to a suitable H5 format compatible with the partitioner. It also modifies the `part.dat` file to ensure the the mesh names are correct.

## Requirements

- GMSH
- HDF5
- MPI
- Python + mpi4py + h5py

## MareNostrum 5 usage

TODO: add the modules that need to be loaded here, can't remember right now. Modify script with them if necessary/convenient.

## Script usage

    "Usage: ./genMesh.sh <geo_file> <elements>(opt) <order>(opt)"

Arguments:

- `geo_file`: The .geo file to use as input, without extension
- `elements`: Number of elements per direction (NxNxN) [optional]
- `order`: Order of the elements [optional]

Both optional arguments must be passed, the script will not work properly if only one of them is used.

## Mesh characteristics

To ensure uniform testing conditions, these are the recommended mesh characteristics:

- Order = 4, 1 partition
  - 2M nodes: 31 elements
  - 4M nodes: 39 elements
  - 8M nodes: 49 elements
  - 12M nodes: 57 elements

Other configurations will be added later.