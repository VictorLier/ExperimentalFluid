from typing import Tuple 
import matplotlib.pyplot as plt
import os

from LDA_calulations import LDA_calcs

class LDA_velocity():
    """ NB! For future improvement each instance of LDA_calcs should be saved in a list and then the list should be iterated over"""
    def __init__(self, main_data_path: str, main_proc_data_path: str) -> None:
        self._main_data_path = main_data_path
        self._main_proc_data_path = main_proc_data_path
        self._folders = self._get_data_folders()
        self.__init_proc_folder()

    def _get_data_folders(self) -> list:
        """ Returns a list of all the folders in the main data path """
        folders = os.listdir(self._main_data_path)
        # sort the folders by name
        sorted_folders = sorted(folders, key=lambda x: int(x.split('_')[-2])) #Here we sort the folders by a chosen method
        # sort folder alphabetically
        # sorted_folders = sorted(folders)
        return sorted_folders
    
    def __init_proc_folder(self) -> None:
        """ Creates a folder in the main_proc_data_path with the same name as the folder in the main_data_path """
        # Check if the folder already exists for every folder in the main_data_path
        for folder in self._folders:
            if not os.path.exists(os.path.join(self._main_proc_data_path, folder)):
                os.makedirs(os.path.join(self._main_proc_data_path, folder))

    def _eval_folders(self) -> None:
        """ Evaluates all the folders in the main_data_path """
        for folder in self._folders:
            lda = LDA_calcs(os.path.join(self._main_data_path, folder), os.path.join(self._main_proc_data_path, folder))
            lda.eval_all_files()
    
    def _remove_outliers(self) -> None:
        """ Removes the outliers from the data """
        for folder in self._folders:
            lda = LDA_calcs(os.path.join(self._main_data_path, folder), os.path.join(self._main_proc_data_path, folder))
            lda.remove_outliers_all_files()
    
    def _get_stats(self) -> None:
        """ Gets the statistics from the data """
        for folder in self._folders:
            lda = LDA_calcs(os.path.join(self._main_data_path, folder), os.path.join(self._main_proc_data_path, folder))
            lda.get_stats_all_files()
    
    def _get_single_mean_velocity(self, folder: str) -> Tuple[float, float]:
        """ Gets the mean velocity from a single folder """
        lda = LDA_calcs(os.path.join(self._main_data_path, folder), os.path.join(self._main_proc_data_path, folder))
        vel, std = lda.get_mean_velocity()
        #print(f"Mean velocity for {folder} is {vel} m/s with a standard deviation of {std} m/s")
        return vel, std

    def _get_mean_velocity_all_folders(self) -> None:
        """ Gets the mean velocity from all the folders in the main_data_path """
        vel = []
        std = []
        for folder in self._folders:
            v, st = self._get_single_mean_velocity(folder)
            vel.append(v)
            std.append(st)
        
        return vel, std

    def full_evaluation(self) -> None:
        """ Evaluates all the folders in the main_data_path """
        self._eval_folders()
        self._remove_outliers()
        self._get_stats()

    def plot_velocities(self) -> None:
        velocities = []
        standard_deviations = []
        for folder in self._folders:
            vel, std = self._get_single_mean_velocity(folder)
            velocities.append(vel)
            standard_deviations.append(std)

        # Plot the velocities wtih error bars
        x_pos = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
        plt.figure()
        #plt.errorbar(range(len(velocities)), velocities, yerr=standard_deviations, fmt='.k', capsize=5)
        plt.errorbar(x_pos, velocities, yerr=standard_deviations, fmt='.k', capsize=5)
        plt.xlabel('Position [cm]')
        plt.ylabel('Velocity [m/s]')
        plt.title('Velocity profile')
        plt.grid()
        plt.ylim(0, 3)
        plt.show()

if __name__ == "__main__":
    parent_dir = os.path.dirname(os.getcwd())
    # data_path = os.path.join(parent_dir, r'LDA_DATA\velocity_profile_data')
    # proc_data_path = os.path.join(parent_dir, r'LDA_DATA\proc_velocity_profile_data')

    data_path = os.path.join(parent_dir, r'LDA_DATA\ke_data')
    proc_data_path = os.path.join(parent_dir, r'LDA_DATA\proc_ke_data')

    lda = LDA_velocity(data_path, proc_data_path)

    if True:
        lda.full_evaluation()
    
    if True:
        #lda._get_mean_velocity_all_folders()
        lda.plot_velocities()

