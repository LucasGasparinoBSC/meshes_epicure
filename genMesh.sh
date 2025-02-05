#!/bin/bash

## Reads a geo file to generate a mesh for SOD2D
## Usage: ./genMesh.sh <geo_file> <elements>(opt) <order>(opt)

# Pass the geo file name as an argument
if [ $# -eq 0 ]; then
    echo "No arguments supplied"
    echo "Usage: ./genMesh.sh <geo_file> <elements>(opt) <order>(opt)"
    exit 1
fi

geo_file=$1.geo
mesh_file=${geo_file%.geo}.msh
echo "Generating mesh for $geo_file"

if [ $# -eq 3 ]; then
    elements=$2
    echo "Number of elements per side: $elements"
    sed -i "2s/.*/n = $elements;/" $geo_file
    order=$3
    echo "Element order: $order"
    sed -i "36s/.*/Mesh.ElementOrder = $order;/" $geo_file
fi

# Check if GMSH is installed/loaded
if ! command -v gmsh &> /dev/null
then
    echo "GMSH could not be found"
    exit 1
fi

# Generate the mesh (assumes a periodic cube)
gmsh $geo_file -o $mesh_file -0

# Check if HDF5 is installed/loaded
if ! command -v h5pcc &> /dev/null
then
    echo "HDF5 could not be found"
    exit 1
fi

# Check if Python is installed/loaded
if ! command -v python &> /dev/null
then
    echo "Python could not be found"
    exit 1
fi

# Check that python has mpi4py and h5py
if ! python -c 'import mpi4py; import h5py' &> /dev/null
then
    echo "Python libraries (mpi4py and h5py) are not installed"
    exit 1
fi

# Convert the mesh to HDF5 format
python gmsh2sod2d.py cube -p 1 -r $order

# Define a symbol holding " as a character
quote='"'

# Modify lines in the part.dat file
sed -i "2s/.*/gmsh_fileName $quote${mesh_file%.msh}$quote /" part.dat
sed -i "4s/.*/mesh_h5_fileName $quote${mesh_file%.msh}$quote /" part.dat