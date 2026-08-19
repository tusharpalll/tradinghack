<?php

require_once __DIR__ . '/../includes/bootstrap.php';
require_login('admin', '/admin/login.php');

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (!verify_csrf($_POST['csrf_token'] ?? null)) {
        set_flash('danger', 'Invalid request token. Please try again.');
        header('Location: /admin/dashboard.php');
        exit;
    }

    $action = $_POST['action'] ?? '';

    if ($action === 'create_client') {
        $name = trim($_POST['name'] ?? '');
        $email = trim($_POST['email'] ?? '');
        $password = $_POST['password'] ?? '';

        if ($name === '' || !filter_var($email, FILTER_VALIDATE_EMAIL) || strlen($password) < 6) {
            set_flash('danger', 'Client details are invalid. Password must be at least 6 characters.');
        } else {
            $existing = db()->prepare('SELECT id FROM users WHERE email = ? LIMIT 1');
            $existing->execute([$email]);
            if ($existing->fetch()) {
                set_flash('danger', 'Email already exists.');
            } else {
                $stmt = db()->prepare('INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)');
                $stmt->execute([$name, $email, password_hash($password, PASSWORD_DEFAULT), 'client']);
                set_flash('success', 'Client created successfully.');
            }
        }
    }

    if ($action === 'create_project') {
        $clientId = (int) ($_POST['client_id'] ?? 0);
        $title = trim($_POST['title'] ?? '');
        $description = trim($_POST['description'] ?? '');
        $status = trim($_POST['status'] ?? 'Not Started');
        $progress = max(0, min(100, (int) ($_POST['progress'] ?? 0)));

        if ($clientId <= 0 || $title === '') {
            set_flash('danger', 'Project title and client are required.');
        } else {
            $stmt = db()->prepare('INSERT INTO projects (client_id, title, description, status, progress) VALUES (?, ?, ?, ?, ?)');
            $stmt->execute([$clientId, $title, $description, $status, $progress]);
            set_flash('success', 'Project created successfully.');
        }
    }

    if ($action === 'update_project') {
        $projectId = (int) ($_POST['project_id'] ?? 0);
        $status = trim($_POST['status'] ?? 'In Progress');
        $progress = max(0, min(100, (int) ($_POST['progress'] ?? 0)));

        if ($projectId <= 0) {
            set_flash('danger', 'Please choose a project to update.');
        } else {
            $stmt = db()->prepare('UPDATE projects SET status = ?, progress = ?, updated_at = NOW() WHERE id = ?');
            $stmt->execute([$status, $progress, $projectId]);
            set_flash('success', 'Project status updated.');
        }
    }

    if ($action === 'add_update') {
        $projectId = (int) ($_POST['project_id'] ?? 0);
        $message = trim($_POST['message'] ?? '');

        if ($projectId <= 0 || $message === '') {
            set_flash('danger', 'Project and update message are required.');
        } else {
            $stmt = db()->prepare('INSERT INTO project_updates (project_id, message) VALUES (?, ?)');
            $stmt->execute([$projectId, $message]);
            db()->prepare('UPDATE projects SET updated_at = NOW() WHERE id = ?')->execute([$projectId]);
            set_flash('success', 'Project update added.');
        }
    }

    header('Location: /admin/dashboard.php');
    exit;
}

$clients = db()->query("SELECT id, name, email FROM users WHERE role = 'client' ORDER BY name ASC")->fetchAll();
$projects = db()->query(
    'SELECT p.id, p.title, p.status, p.progress, p.updated_at, u.name AS client_name
     FROM projects p
     INNER JOIN users u ON u.id = p.client_id
     ORDER BY p.updated_at DESC, p.id DESC'
)->fetchAll();

require __DIR__ . '/../includes/header.php';
?>
<h1 class="h3 mb-4">Admin Dashboard</h1>

<div class="row g-4">
    <div class="col-lg-6">
        <div class="card shadow-sm h-100">
            <div class="card-body">
                <h2 class="h5">Create Client</h2>
                <form method="post" class="row g-2">
                    <input type="hidden" name="csrf_token" value="<?= e(csrf_token()) ?>">
                    <input type="hidden" name="action" value="create_client">
                    <div class="col-12"><input class="form-control" name="name" placeholder="Client name" required></div>
                    <div class="col-12"><input class="form-control" type="email" name="email" placeholder="Client email" required></div>
                    <div class="col-12"><input class="form-control" type="password" name="password" placeholder="Temporary password (min 6)" minlength="6" required></div>
                    <div class="col-12"><button class="btn btn-dark" type="submit">Create Client</button></div>
                </form>
            </div>
        </div>
    </div>

    <div class="col-lg-6">
        <div class="card shadow-sm h-100">
            <div class="card-body">
                <h2 class="h5">Create Project</h2>
                <form method="post" class="row g-2">
                    <input type="hidden" name="csrf_token" value="<?= e(csrf_token()) ?>">
                    <input type="hidden" name="action" value="create_project">
                    <div class="col-12">
                        <select class="form-select" name="client_id" required>
                            <option value="">Select client</option>
                            <?php foreach ($clients as $client): ?>
                                <option value="<?= (int) $client['id'] ?>"><?= e($client['name']) ?> (<?= e($client['email']) ?>)</option>
                            <?php endforeach; ?>
                        </select>
                    </div>
                    <div class="col-12"><input class="form-control" name="title" placeholder="Project title" required></div>
                    <div class="col-12"><textarea class="form-control" name="description" placeholder="Description" rows="2"></textarea></div>
                    <div class="col-md-7"><input class="form-control" name="status" value="Not Started" required></div>
                    <div class="col-md-5"><input class="form-control" type="number" min="0" max="100" name="progress" value="0" required></div>
                    <div class="col-12"><button class="btn btn-dark" type="submit">Create Project</button></div>
                </form>
            </div>
        </div>
    </div>

    <div class="col-lg-6">
        <div class="card shadow-sm h-100">
            <div class="card-body">
                <h2 class="h5">Update Project Status</h2>
                <form method="post" class="row g-2">
                    <input type="hidden" name="csrf_token" value="<?= e(csrf_token()) ?>">
                    <input type="hidden" name="action" value="update_project">
                    <div class="col-12">
                        <select class="form-select" name="project_id" required>
                            <option value="">Select project</option>
                            <?php foreach ($projects as $project): ?>
                                <option value="<?= (int) $project['id'] ?>"><?= e($project['title']) ?> - <?= e($project['client_name']) ?></option>
                            <?php endforeach; ?>
                        </select>
                    </div>
                    <div class="col-md-7"><input class="form-control" name="status" placeholder="Status (In Progress, Testing...)" required></div>
                    <div class="col-md-5"><input class="form-control" type="number" min="0" max="100" name="progress" placeholder="Progress %" required></div>
                    <div class="col-12"><button class="btn btn-primary" type="submit">Update Status</button></div>
                </form>
            </div>
        </div>
    </div>

    <div class="col-lg-6">
        <div class="card shadow-sm h-100">
            <div class="card-body">
                <h2 class="h5">Add Project Update</h2>
                <form method="post" class="row g-2">
                    <input type="hidden" name="csrf_token" value="<?= e(csrf_token()) ?>">
                    <input type="hidden" name="action" value="add_update">
                    <div class="col-12">
                        <select class="form-select" name="project_id" required>
                            <option value="">Select project</option>
                            <?php foreach ($projects as $project): ?>
                                <option value="<?= (int) $project['id'] ?>"><?= e($project['title']) ?> - <?= e($project['client_name']) ?></option>
                            <?php endforeach; ?>
                        </select>
                    </div>
                    <div class="col-12"><textarea class="form-control" name="message" rows="3" placeholder="Latest status note / update" required></textarea></div>
                    <div class="col-12"><button class="btn btn-primary" type="submit">Add Update</button></div>
                </form>
            </div>
        </div>
    </div>
</div>

<div class="card shadow-sm mt-4">
    <div class="card-body">
        <h2 class="h5">Recent Projects</h2>
        <div class="table-responsive">
            <table class="table align-middle">
                <thead>
                    <tr>
                        <th>Project</th>
                        <th>Client</th>
                        <th>Status</th>
                        <th>Progress</th>
                        <th>Updated</th>
                    </tr>
                </thead>
                <tbody>
                    <?php foreach ($projects as $project): ?>
                        <tr>
                            <td><?= e($project['title']) ?></td>
                            <td><?= e($project['client_name']) ?></td>
                            <td><?= e($project['status']) ?></td>
                            <td><?= (int) $project['progress'] ?>%</td>
                            <td><?= e($project['updated_at']) ?></td>
                        </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
        </div>
    </div>
</div>

<?php require __DIR__ . '/../includes/footer.php'; ?>
