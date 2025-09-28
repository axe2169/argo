// Default filter: show all
filterServices("all");

function filterServices(category) {
  const items = document.getElementsByClassName("filterDiv");
  for (let i = 0; i < items.length; i++) {
    items[i].classList.remove("show");
    if (category === "all" || items[i].classList.contains(category)) {
      items[i].classList.add("show");
    }
  }

  // Update button active state
  const buttons = document.getElementsByClassName("btn");
  for (let i = 0; i < buttons.length; i++) {
    buttons[i].classList.remove("active");
  }
  document.querySelector(`button[onclick="filterServices('${category}')"]`).classList.add("active");
}

