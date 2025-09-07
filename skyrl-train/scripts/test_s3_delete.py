import fsspec

p = "s3://skyrl-test/ckpts/global_step_2/"

fs = fsspec.filesystem("s3")
print("exists(no slash):", fs.exists(p))
print("exists(with slash):", fs.exists(p + "/"))
print("isdir(no slash):", fs.isdir(p))
print("isdir(with slash):", fs.isdir(p + "/"))

print("ls(no slash):")
try:
    print(fs.ls(p))
except Exception as e:
    print("ls failed:", e)

print("ls(with slash):")
try:
    print(fs.ls(p + "/"))
except Exception as e:
    print("ls failed:", e)

# CAREFUL: uncomment to actually test deletion
# print("rm(no slash, recursive=False):")
# try:
#     fs.rm(p, recursive=False)
#     print("rm done")
# except Exception as e:
#     print("rm failed:", e)

print("rm(with slash, recursive=True):")
try:
    fs.rm(p + "/", recursive=True)
    print("rm done")
except Exception as e:
    print("rm failed:", e)
