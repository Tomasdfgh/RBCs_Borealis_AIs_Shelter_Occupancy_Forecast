<img width="1437" alt="Screenshot 2024-04-24 at 6 30 50 PM" src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/105636722/2f893bee-4825-437b-9fda-5dacf9281ac7">

<br></br>

## Shelter Sight User Interface
### Background
In our journey to create a reliable model, we've dedicated ourselves to crafting a prototype that leverages data collected from the city of Toronto, paired with our models. This user interface (UI) prototype isn't just the final output of our project; it also serves as an illustration of how our data and models can be put into practical use.

This UI prototype represents the culmination of countless hours of research, development, and fine-tuning. It's a tangible manifestation of our commitment to delivering impactful solutions.

### Video Demonstration
The video below provides a demonstration to how the software functions. We will also go over all the functionalities in this readme as well, but this video serves as an alternative way of understanding how ShelterSight works. 

<div align="center">
    <a href="https://www.youtube.com/watch?v=-DDK9pMYjrk">
    <img src="https://img.youtube.com/vi/-DDK9pMYjrk/0.jpg" alt="DEMO VIDEO" style="width:55%;">
    <p>Shelter Sight Demo</p>
  </a>
</div>

### Shelter Selection

In order to view any information about any shelters, you will have to select the program ID of the shelters in the dropdown of Shelter Selection. Once your shelter have been selected, click on the add button. The Selected Shelters count will increment by 1 when the shelter has been added.

### Shelter Information

This section will display the information of the shelters that you have selected. Click on the Program ID's dropdown, and the options available are the Program IDs of all the shelters that you have selected. Once you clicked on any of the Program IDs, its information will show up in the rest of the sections. Infomration that is viewable in this section incldues the Organization name, Shelter Group, Location Name, Location Address, Postal Code, and Capacity Type. This section is important as the chosen shelter will be the shelter displayed in the "Output" section and the second graph type as well (more information on that in the "Change and View Graph" section.

### Remove and Reset Functionalities

To remove a shelter that you have chosen in the Shelter Selection section, you can select that shelter in the Shelter Information's dropdown and click the remove button. That will remove the singular shelter that you have selected and decrease the selected shelters count by 1. You also have the option to reset everything by clicking on the reset button. This will clear all selected shelters and the graph and the Forecast Period dates.

### Forecast Period

Forecast Period is a functionality for the user to choose a specific window of time to look at in the graph. When you have chosen your shelters without a forecast period, you can view every bit of data that is available (which includes all the actual data that has come to pass of every shelters and also their predicted data for the next 60 days as well). When you click on Open Calender for the Forecast Period, a popup will appear with two calendars: one for the start of the window, and one for the end. Once the two dates have been selected, you can click on forecast again, and the graph will shrink to the window that you have selected.

### Change and View Graph, and Download Data

### Output and Select Model
