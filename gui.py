import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image
from test import load_model, apply_counting

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("green")

model_path_1 = 'Trained_models/01-23_14-53_SHHB_Res50_1e-05/best_model.pth'
model_path_2 = 'Trained_models/01-24_14-13_SHHB_Res50_1e-05/best_model.pth'
model_path_3 = 'Trained_models/02-05_18-46_SHHA_Res50_1e-05/best_model.pth'
model_path_4 = 'Trained_models/02-06_15-12_SHHM_Res50_1e-05/best_model.pth'

net1 = load_model(model_path_1)
net2 = load_model(model_path_2)
net3 = load_model(model_path_3)
net4 = load_model(model_path_4)

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Crowd Counting Application")
        self.geometry("900x600")
        self.img_frame_size_big = 400
        self.img_frame_size_small = 250
        self.factor_big = 1
        self.factor_small = 1
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)
        self.grid_columnconfigure((0, 1, 2, 3), weight=1)

        empty_img_big = Image.new("RGBA", (self.img_frame_size_big, self.img_frame_size_big), (0, 0, 0, 0))
        empty_img_small = Image.new("RGBA", (self.img_frame_size_small, self.img_frame_size_small), (0, 0, 0, 0))
        self.selected_img = Image.new("RGB", (self.img_frame_size_big, self.img_frame_size_small), (0, 0, 0))

        self.image_frame_1 = ctk.CTkFrame(self)
        self.image_frame_2 = ctk.CTkFrame(self)
        self.image_frame_3 = ctk.CTkFrame(self)
        self.image_frame_4 = ctk.CTkFrame(self)

        self.image_frame_1.grid_rowconfigure((0, 1), weight=1)
        self.image_frame_2.grid_rowconfigure((0, 1), weight=1)
        self.image_frame_3.grid_rowconfigure((0, 1), weight=1)
        self.image_frame_4.grid_rowconfigure((0, 1), weight=1)

        self.image_frame_1.grid_columnconfigure(0, weight=1)
        self.image_frame_2.grid_columnconfigure(0, weight=1)
        self.image_frame_3.grid_columnconfigure(0, weight=1)
        self.image_frame_4.grid_columnconfigure(0, weight=1)

        self.image_frame_1.grid(row=0, column=2, padx=10, pady=(10, 10), rowspan=2, sticky="nsew")
        self.image_frame_2.grid(row=2, column=2, padx=10, pady=(0, 10), rowspan=2, sticky="nsew")
        self.image_frame_3.grid(row=0, column=3, padx=10, pady=(10, 10), rowspan=2, sticky="nsew")
        self.image_frame_4.grid(row=2, column=3, padx=10, pady=(0, 10), rowspan=2, sticky="nsew")

        self.image_0 = ctk.CTkImage(light_image=empty_img_big, size=(self.img_frame_size_big, self.img_frame_size_big))
        self.image_1 = ctk.CTkImage(light_image=empty_img_small,
                                    size=(self.img_frame_size_small, self.img_frame_size_small))
        self.image_2 = ctk.CTkImage(light_image=empty_img_small,
                                    size=(self.img_frame_size_small, self.img_frame_size_small))
        self.image_3 = ctk.CTkImage(light_image=empty_img_small,
                                    size=(self.img_frame_size_small, self.img_frame_size_small))
        self.image_4 = ctk.CTkImage(light_image=empty_img_small,
                                    size=(self.img_frame_size_small, self.img_frame_size_small))

        self.image_label_0 = ctk.CTkLabel(self, text="", image=self.image_0)
        self.image_label_1 = ctk.CTkLabel(master=self.image_frame_1, text="", image=self.image_1)
        self.image_label_2 = ctk.CTkLabel(master=self.image_frame_2, text="", image=self.image_2)
        self.image_label_3 = ctk.CTkLabel(master=self.image_frame_3, text="", image=self.image_3)
        self.image_label_4 = ctk.CTkLabel(master=self.image_frame_4, text="", image=self.image_4)

        self.image_label_0.grid(row=1, column=0, pady=10, padx=20, sticky="new", rowspan=3, columnspan=2)
        self.image_label_1.grid(row=0, column=0, pady=10, padx=10, sticky="ew")
        self.image_label_2.grid(row=0, column=0, pady=10, padx=10, sticky="ew")
        self.image_label_3.grid(row=0, column=0, pady=10, padx=10, sticky="ew")
        self.image_label_4.grid(row=0, column=0, pady=10, padx=10, sticky="ew")

        self.text_pred_1 = ctk.CTkLabel(master=self.image_frame_1, text="")
        self.text_pred_2 = ctk.CTkLabel(master=self.image_frame_2, text="")
        self.text_pred_3 = ctk.CTkLabel(master=self.image_frame_3, text="")
        self.text_pred_4 = ctk.CTkLabel(master=self.image_frame_4, text="")

        self.text_pred_1.grid(row=1, column=0, padx=10, pady=10, sticky="new")
        self.text_pred_2.grid(row=1, column=0, padx=10, pady=10, sticky="new")
        self.text_pred_3.grid(row=1, column=0, padx=10, pady=10, sticky="new")
        self.text_pred_4.grid(row=1, column=0, padx=10, pady=10, sticky="new")

        self.button_select = ctk.CTkButton(self, text="select image", command=self.select_image)
        self.button_select.grid(row=0, column=0, padx=10, pady=(10, 30), sticky="ew", columnspan=1)
        self.button_apply = ctk.CTkButton(self, text="count people", command=self.apply_model)
        self.button_apply.grid(row=0, column=1, padx=10, pady=(10, 30), sticky="ew", columnspan=1)


    def select_image(self):
        file_path = filedialog.askopenfilename(title="Bild auswählen", filetypes=[("Bilder", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.display_image(file_path)

    def display_image(self, file_path):
        img = Image.open(file_path)
        if(img.width > img.height):
            self.factor_big = self.img_frame_size_big / img.width
            self.factor_small = self.img_frame_size_small / img.width
        else:
            self.factor_big = self.img_frame_size_big / img.height
            self.factor_small = self.img_frame_size_small / img.height

        self.image_0.configure(light_image=img, size=(img.width * self.factor_big, img.height * self.factor_big))
        self.image_label_0.configure(image=self.image_0)
        self.selected_img = img

    def apply_model(self):
        img_1, pred_1 = apply_counting(self.selected_img, net1)
        img_2, pred_2 = apply_counting(self.selected_img, net2)
        img_3, pred_3 = apply_counting(self.selected_img, net3)
        img_4, pred_4 = apply_counting(self.selected_img, net4)

        self.image_1.configure(light_image=img_1, size=(self.selected_img.width * self.factor_small,
                                                        self.selected_img.height * self.factor_small))
        self.image_label_1.configure(image=self.image_1)
        self.text_pred_1.configure(text=f"prediction Res50 - SHHB v1: \n{pred_1} people")

        self.image_2.configure(light_image=img_2, size=(self.selected_img.width * self.factor_small,
                                                        self.selected_img.height * self.factor_small))
        self.image_label_2.configure(image=self.image_2)
        self.text_pred_2.configure(text=f"prediction Res50 - SHHB v2: \n{pred_2} people")

        self.image_3.configure(light_image=img_3, size=(self.selected_img.width * self.factor_small,
                                                        self.selected_img.height * self.factor_small))
        self.image_label_3.configure(image=self.image_3)
        self.text_pred_3.configure(text=f"prediction Res50 - SHHA: \n{pred_3} people")

        self.image_4.configure(light_image=img_4, size=(self.selected_img.width * self.factor_small,
                                                        self.selected_img.height * self.factor_small))
        self.image_label_4.configure(image=self.image_4)
        self.text_pred_4.configure(text=f"prediction Res50 - SHHM: \n{pred_4} people")



if __name__ == "__main__":
    app = App()
    app.mainloop()