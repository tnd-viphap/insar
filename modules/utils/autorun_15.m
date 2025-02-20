% Get all folders containing 'PATCH' in their name
allFolders = dir();
folders = {allFolders([allFolders.isdir]).name};
folders = folders(contains(folders, 'PATCH'));

% Loop through each folder
for i = 1:length(folders)
    folder = folders{i};
    
    % Change to the folder
    if exist(folder, 'dir')
        fprintf('Processing folder: %s\n', folder);
        cd(folder);
        
        % Run your sequence of commands here
        % Example: Run a MATLAB script or function
        % myFunction();
        % run('myscript.m');
        
        % Change back to the original directory
        cd ..
    else
        fprintf('Folder not found: %s\n', folder);
    end
end
