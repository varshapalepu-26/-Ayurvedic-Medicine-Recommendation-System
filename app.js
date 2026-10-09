document.addEventListener('DOMContentLoaded', () => {
  const form   = document.getElementById('medicineForm');
  const output = document.getElementById('output');

  // Full list of symptoms (from your previous checkboxes)
  const symptoms = [
    "itching","skin_rash","nodal_skin_eruptions","continuous_sneezing","shivering","chills","joint_pain",
    "stomach_pain","acidity","ulcers_on_tongue","muscle_wasting","vomiting","burning_micturition",
    "spotting_urination","fatigue","weight_gain","anxiety","cold_hands_and_feets","mood_swings",
    "weight_loss","restlessness","lethargy","patches_in_throat","irregular_sugar_level","cough",
    "high_fever","sunken_eyes","breathlessness","sweating","dehydration","indigestion","headache",
    "yellowish_skin","dark_urine","nausea","loss_of_appetite","pain_behind_the_eyes","back_pain",
    "constipation","abdominal_pain","diarrhoea","mild_fever","yellow_urine","yellowing_of_eyes",
    "acute_liver_failure","fluid_overload","swelling_of_stomach","swelled_lymph_nodes","malaise",
    "blurred_and_distorted_vision","phlegm","throat_irritation","redness_of_eyes","sinus_pressure",
    "runny_nose","congestion","chest_pain","weakness_in_limbs","fast_heart_rate","pain_during_bowel_movements",
    "pain_in_anal_region","bloody_stool","irritation_in_anus","neck_pain","dizziness","cramps",
    "bruising","obesity","swollen_legs","swollen_blood_vessels","puffy_face_and_eyes","enlarged_thyroid",
    "brittle_nails","swollen_extremeties","excessive_hunger","extra_marital_contacts",
    "drying_and_tingling_lips","slurred_speech","knee_pain","hip_joint_pain","muscle_weakness",
    "stiff_neck","swelling_joints","movement_stiffness","spinning_movements","loss_of_balance",
    "unsteadiness","weakness_of_one_body_side","loss_of_smell","bladder_discomfort",
    "foul_smell_of_urine","continuous_feel_of_urine","passage_of_gases","internal_itching",
    "toxic_look_(typhos)","depression","irritability","muscle_pain","altered_sensorium",
    "red_spots_over_body","belly_pain","abnormal_menstruation","dischromic_patches",
    "watering_from_eyes","increased_appetite","polyuria","family_history","mucoid_sputum",
    "rusty_sputum","lack_of_concentration","visual_disturbances","receiving_blood_transfusion",
    "receiving_unsterile_injections","coma","stomach_bleeding","distention_of_abdomen",
    "history_of_alcohol_consumption","blood_in_sputum","prominent_veins_on_calf","palpitations",
    "painful_walking","pus_filled_pimples","blackheads","scurring","skin_peeling",
    "silver_like_dusting","small_dents_in_nails","inflammatory_nails","blister","red_sore_around_nose",
    "yellow_crust_ooze"
  ];

  const symptomGroup = document.getElementById('symptomGroup');
  const searchBox = document.getElementById('searchSymptoms');

  // Sort symptoms alphabetically and render them
  symptoms.sort((a, b) => a.localeCompare(b)).forEach(symptom => {
    const label = document.createElement('label');
    label.innerHTML = `<input type="checkbox" name="symptom" value="${symptom}"> ${symptom.replace(/_/g,' ').replace(/\b\w/g, c => c.toUpperCase())}`;
    symptomGroup.appendChild(label);
  });

  // Live filter symptoms on search
  searchBox.addEventListener('input', () => {
    const filter = searchBox.value.toLowerCase();
    const labels = symptomGroup.querySelectorAll('label');
    labels.forEach(label => {
      const text = label.textContent.toLowerCase();
      label.style.display = text.includes(filter) ? '' : 'none';
    });
  });

  form.addEventListener('submit', async e => {
    e.preventDefault();

    // Gather selected symptoms
    const selectedSymptoms = Array.from(
      form.querySelectorAll('input[name="symptom"]:checked')
    ).map(cb => cb.value);

    const payload = {
      symptoms: selectedSymptoms,
      age: form.age.value,
      gender: form.gender.value,
      severity: form.severity.value
    };

    try {
      const res  = await fetch('/predict', {
        method:  'POST',
        headers: {'Content-Type': 'application/json'},
        body:    JSON.stringify(payload)
      });
      const data = await res.json();

      output.style.display = 'block';
      output.innerHTML = `
        <p><strong>Selected Symptoms:</strong></p>
        <ul>${selectedSymptoms.map(s => `<li>${s}</li>`).join('')}</ul>
        <p><strong>Disease:</strong> ${data.disease}</p>
        <p><strong>Medicines:</strong></p>
        <ol>${data.medicines.map(m => `<li>${m}</li>`).join('')}</ol>
        <button id="closeBtn" style="width:100px;position:absolute; bottom:10px;background:green; color:white; border:none;padding-top:5px; border-radius:5px;cursor:pointer;">Close</button>
      `;
      document.getElementById("closeBtn").addEventListener("click", () => {
        output.classList.remove("show");
        setTimeout(() => output.style.display = "none", 500);
      });

    } catch (err) {
      console.error(err);
      alert('Prediction failed.');
    }
  });
});
