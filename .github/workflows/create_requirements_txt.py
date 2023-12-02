import os.path

with open(os.path.dirname(__file__) + '/../../dependencies.md', 'r') as f:
    contents = f.read().split('\n')

python_package_idx = contents.index('## Python Packages')+1
end_of_python_packages : int

for n, line in enumerate(contents[python_package_idx:]):
    if '#' in line:
        end_of_python_packages = n
        break

packages = contents[python_package_idx:end_of_python_packages+python_package_idx]

with open(os.path.dirname(__file__) + '/requirements.txt', 'w') as f:
    for package in packages:
        package = package.strip('- ')
        f.write(package + '\n')