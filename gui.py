import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image
from test import apply_counting

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("green")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        num_col = 2
        self.img_frame_size = 400
        self.factor = 1
        self.title("Crowd Counting Application")
        self.geometry("900x600")
        self.grid_rowconfigure((0, 1, 2), weight=1)
        self.grid_columnconfigure((0, 1), weight=1)

        empty_img = Image.new("RGBA", (self.img_frame_size, self.img_frame_size), (0, 0, 0, 0))
        self.selected_img = Image.new("RGB", (self.img_frame_size, self.img_frame_size), (0, 0, 0))

        self.image_1 = ctk.CTkImage(light_image=empty_img, size=(self.img_frame_size, self.img_frame_size))
        self.image_2 = ctk.CTkImage(light_image=empty_img, size=(self.img_frame_size, self.img_frame_size))
        self.image_label_1 = ctk.CTkLabel(self, text="", image=self.image_1)
        self.image_label_1.grid(row=1, column=0, padx=0, pady=(0, 10), sticky="nsew", columnspan=1)
        self.image_label_2 = ctk.CTkLabel(self, text="", image=self.image_2)
        self.image_label_2.grid(row=1, column=1, padx=0, pady=(0, 10), sticky="nsew", columnspan=1)

        self.text_pred = ctk.CTkLabel(self, text="")
        self.text_pred.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

        self.button_select = ctk.CTkButton(self, text="select image", command=self.select_image)
        self.button_select.grid(row=0, column=0, padx=10, pady=10, sticky="new", columnspan=1)
        self.button_apply = ctk.CTkButton(self, text="count people", command=self.apply_model)
        self.button_apply.grid(row=0, column=1, padx=10, pady=10, sticky="new", columnspan=1)


    def select_image(self):
        file_path = filedialog.askopenfilename(title="Bild auswählen", filetypes=[("Bilder", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.display_image(file_path)

    def display_image(self, file_path):
        img = Image.open(file_path)
        if(img.width > img.height):
            self.factor = self.img_frame_size / img.width
        else:
            self.factor = self.img_frame_size / img.height

        self.image_1.configure(light_image=img, size=(img.width * self.factor, img.height * self.factor))
        self.image_label_1.configure(image=self.image_1)
        self.selected_img = img

    def apply_model(self):
        img, pred = apply_counting(self.selected_img)
        self.image_2.configure(light_image=img, size=(self.selected_img.width * self.factor,
                               self.selected_img.height * self.factor))
        self.image_label_2.configure(image=self.image_2)

        self.text_pred.configure(text=f"prediction: {pred} people")



if __name__ == "__main__":
    app = App()
    app.mainloop()